# Acceleration Method Review

Reviewer notes on the GPU acceleration of Ordinal/Z-Score SuStaIn in this
repository, against Sountsov, Carroll, & Hoffman (2024), *Running Markov
Chain Monte Carlo on Modern Hardware and Software*
(arXiv:2411.04260v1), and adjacent work.

## 1. What the paper actually recommends

The headline message of Sountsov et al. is **vectorise across chains**, not
across data. On modern GPUs/TPUs, a single MCMC chain leaves most of the
device idle; the natural unit of parallelism is the chain. The
recommended pattern is roughly:

1. Express the log-likelihood as a JIT-compiled, side-effect-free function.
2. `vmap` (JAX) / `torch.vmap` / batch dimension (PyTorch) it over a
   "chains" axis with thousands of independent walkers.
3. Adapt the sampler using cross-chain statistics (ChEES-HMC, MEADS,
   SNAPER-HMC) so warm-up is short.
4. Drive the whole MCMC loop on device — `jax.lax.scan` / a Torch loop
   that never copies state back to host until the run finishes.
5. Use float32 in production. The precision loss is normally absorbed by
   MCMC noise; reserve float64 for validation runs.

The paper's anti-patterns are exactly what bottleneck a naive port:

- Per-iteration host↔device synchronisation.
- Python control flow that triggers `.item()` / `.cpu().numpy()` inside
  the inner loop.
- Single chains that under-utilise the device.
- Custom CUDA kernels (NumPy/PyTorch primitives are usually faster than
  hand-written ones because they compose into fused graphs).

The samplers it discusses in detail are continuous (HMC, NUTS, MALT,
slice). It does **not** address the case here directly:
constrained-permutation Metropolis–Hastings on a discrete state space.
That gap matters when applying its recommendations to SuStaIn — see §3.

## 2. What the repository does

The repository implements two GPU strategies in parallel and has
explicitly chosen the second:

**v1 (deprecated, but the files are still present).**
`pySuStaIn/torch_mcmc.py` runs the whole MCMC loop on GPU. It also
batches 128 proposals per step from the same state.

- The 128-from-the-same-state pattern is a different sampler, not a
  faster version of the same one — the documentation in
  `GPU_ACCELERATION_SUMMARY.md:11-17` already calls this out.
- The on-device loop is littered with `.item()` calls
  (`torch_mcmc.py:83, 84, 157, 200, 213, 214`) and Python lists
  (`torch_mcmc.py:217-220`). Each one forces a CUDA sync and stalls the
  pipeline, so the implementation is almost certainly slower than CPU
  on small problems. This is exactly the anti-pattern Sountsov et al.
  warn about.

**v2 (current).**
`pySuStaIn/TorchOrdinalSustain.py` overrides only
`_calculate_likelihood_stage` and `_calculate_likelihood`
(lines 123-183) and leaves the MCMC loop on CPU. Each call:

1. Wraps `sustainData` as a Torch tensor pair (cached by `id()` —
   `TorchOrdinalSustain.py:99-121`).
2. Sends the small sequence `S` to device
   (`TorchOrdinalSustain.py:140`).
3. Does one big gather + sum + exp on device using a pre-built index
   matrix (`torch_likelihood.py:444-546`).
4. Copies the M × (N+1) result back to host
   (`TorchOrdinalSustain.py:144`).

The gather-index construction is genuinely clever:
`TorchOrdinalLikelihoodCalculator._calculate_likelihood_stage_torch`
(`torch_likelihood.py:444-546`) replaces the per-stage Python loop with
a single advanced-indexing kernel by precomputing which columns of
`[log(prob_score) | log(prob_nl)]` to sum for each of the N+1 stages.
That collapses ~22 small kernel launches into one large one — the
dominant constant-factor improvement over a naive PyTorch port.

## 3. Validity

The v2 acceleration is **valid**.

- It changes only how the likelihood is *computed*, not the proposal
  distribution, accept/reject step, RNG sequence, or invariant measure.
- With `force_float64=True` the validation harness reports 0.00e+00 max
  difference vs CPU on 25 random sequences across N_S∈{1,2}
  (`GPU_ACCELERATION_SUMMARY.md:81-88`).
- The subset-handling fix (respecting the `sustainData` argument
  rather than reading from cached GPU state) restores correctness for
  cross-validation and multi-subtype runs.

So the current method is a correct, defensible drop-in. But it is also
the **opposite** of what Sountsov et al. recommend: it vectorises across
data (M subjects within one chain) rather than across chains. SuStaIn
typically has M ≈ 5,000 and N ≈ 22 — large M is helpful, but a T4 GPU
can comfortably hold 1,000 such chains in memory at once and the device
is being left idle between iterations.

The expected speedup at M = 5,000 on a T4 is probably in the 5-15x range
once host↔device sync is accounted for (see §4.1), not the 10-20x
optimistically quoted in `GPU_ORDINAL_OPTIMIZATION.md:158-168` — those
numbers were extrapolated from Z-score benchmarks where the per-call
work is larger.

## 4. Concrete opportunities to optimise

Ranked by expected impact on end-to-end MCMC wall time.

### 4.1 Eliminate per-iteration host↔device sync (high impact)

The current `_perform_mcmc` (`OrdinalSustain.py:366-467`) runs on CPU
and calls `_calculate_likelihood` once per iteration. Each call does
one `to_torch(S)` (host→device) and one `to_numpy(p_perm_k)`
(device→host) — roughly 100,000 sync pairs per MCMC chain. Each sync
stalls the pipeline.

The fix is structural, not algorithmic. Two options:

- **Keep things on device but return only the scalar log-likelihood.**
  CPU MCMC only needs `loglike` for the accept/reject test; the full
  `p_perm_k` is only consumed inside `_calculate_likelihood` and the
  staging step at the end. If the override returns a single float per
  call, the host transfer is one float, not M × (N+1) doubles. The
  staging step at the end can rebuild p_perm_k from the final samples.

- **Push the inner loop on device.** The accept/reject test is a single
  comparison; the proposal logic is the only thing that doesn't
  vectorise cleanly (see §4.3). A scripted Torch loop that does
  proposal → likelihood → MH on device, with the constraint logic
  precomputed once and replayed on GPU, would remove the sync entirely.
  This is more invasive — but it is also a prerequisite for §4.2.

Estimated speedup on top of v2: **2-5x**, larger as M shrinks.

### 4.2 Vectorise across chains (high impact — the paper's main point)

Run K independent chains as one batched GPU computation. The natural
batching axis is the leading dimension everywhere:

- `S` becomes `(K, N)` instead of `(N,)`.
- `p_perm_k` becomes `(K, M, N+1)`.
- `gather_indices` becomes `(K, N+1, max_cols)`.
- The accept/reject test compares K independent `loglike` values to K
  independent Uniform draws — no synchronisation needed across chains.

For SuStaIn the chains can share the data tensors (`prob_nl`,
`prob_score`) — they read-only — so memory cost is `K × M × (N+1) ×
4 bytes` for `p_perm_k`. At M=5,000, N=22, K=512, that is ~240 MB:
fits on a T4 with room to spare.

The harder part is the proposal logic: each chain needs an independent
move under biomarker-ordering constraints. The structure of those
constraints is fixed (it depends only on `stage_biomarker_index`,
`stage_score`), so the valid-position lookup for each (subtype, event)
pair can be precomputed into a `(N, max_positions)` lookup table and
indexed per chain.

This is the change with the largest expected payoff. Combined with §4.1
it should give a further **10-50x** wall-time reduction over what v2
achieves today, because device utilisation goes from ~5% to near 100%.

Reasonable target: 1,000 SuStaIn MCMC chains on a single T4, finishing
in comparable wall time to 1 chain on CPU.

### 4.3 Incremental likelihood updates (medium impact, independent of GPU)

A single MCMC step moves one event in one subtype's sequence from
position `i` to position `j`. The current code recomputes all N+1
stages of `p_perm_k`, but only the columns `min(i,j)+1 .. max(i,j)` can
have changed. Typical proposals move by ≤2 positions, so on average
20+ of 23 stages are recomputed for nothing.

Implementing incremental updates would reduce per-iteration work by
roughly an order of magnitude regardless of CPU/GPU, but it requires
keeping the per-stage product as a running state rather than rebuilding
each call. This is straightforward in log-space (subtract the old
column, add the new) but it does change the numerical exposure — the
running log-sum can drift and would need periodic refresh.

Estimated speedup: **5-15x**, multiplicative with §4.1/§4.2.

### 4.4 Fix the float32 underflow guard (correctness bug)

`safe_torch_operations(tensor, 'log')` adds `1e-250` before taking the
log (`torch_backend.py:282`). In float32 (the default GPU dtype, set in
`torch_backend.py:39`) the smallest positive normal is ~1.18e-38, and
the next value below it is ~1.4e-45 subnormal. Anything below ~6e-39
is unrepresentable. So `1e-250 + x` in float32 is exactly `x`, and the
guard is silently a no-op.

In practice this only matters when `total_prob_subj` underflows to 0,
which can happen for a subject far from any stage in an early MCMC
step. Concretely, the current code can return `log(0) = -inf`, which
propagates through the MH ratio and causes the proposal to be rejected
regardless of merit (or accepted regardless of merit if both states
underflow). Either:

- Carry the computation in log-space and use `logsumexp` instead of
  `log(sum(exp(...)))`, **or**
- Add the offset *after* casting to float64 for the final log, **or**
- Compute the log-likelihood per subject as `logsumexp` of the per-stage
  log-likelihoods (which is already partially available in
  `torch_likelihood.py:533-541` where `log_combined` lives).

The third option is cleanest because `log_combined` already exists.

### 4.5 The CPU "parallel MCMC" path is not what it claims

`pySuStaIn/parallel_mcmc.py` advertises 2-4x speedup from running
multiple chains, but:

- It uses `ThreadPoolExecutor` (`parallel_mcmc.py:205`), which means
  Python GIL-bound threads. SuStaIn's likelihood loop is pure
  NumPy/Python with no released GIL, so the threads run serially.
- The fallback `_run_simplified_mcmc` (`parallel_mcmc.py:165-199`)
  returns mock samples with `rng.random()` as the likelihood, so any
  benchmark that hits this path is meaningless.
- It sets `np.random.seed(seed)` in each thread
  (`parallel_mcmc.py:127`), mutating global state from multiple threads.

If multi-chain CPU parallelism is wanted as a fallback for systems
without a GPU, use `multiprocessing.Pool` with proper pickling (each
worker holds its own SuStaIn instance) — or just rely on §4.2 once it
exists.

### 4.6 NumPyro path uses the wrong sampler family

`pySuStaIn/numpyro_sustain.py` reaches for `NUTS`/`HMC`
(`numpyro_sustain.py:12`). SuStaIn's state space is a constrained
permutation, not a continuous Euclidean space. NUTS would either
require a continuous relaxation (e.g. Plackett-Luce parameterisation
with auxiliary continuous variables) or a Mixed-HMC variant.

If JAX/NumPyro is in scope for the thesis at all, the right move is to
implement the SuStaIn proposal as a custom `numpyro.infer.MCMCKernel`
(or BlackJAX kernel) so that `numpyro.infer.MCMC.run` can vmap it
across chains automatically. This is essentially a clean
implementation of §4.2 in a maintained framework, and gets you
trivially-parallel multi-chain MCMC, `jit`, `lax.scan`, and proper
diagnostics (Nested-Rhat — Margossian et al., arXiv:2110.13017) for
free. Strongly recommended over hand-rolling a Torch multi-chain loop
if the calendar allows.

### 4.7 Minor / cosmetic

- `_ensure_torch_data` caches by `id(sustainData)`
  (`TorchOrdinalSustain.py:103`). If two distinct `OrdinalSustainData`
  objects happen to be allocated at the same address after garbage
  collection, the cached wrapper points at stale data. Mitigated in
  practice by the size check on line 107, but a weak-reference cache or
  a hash of `(prob_nl.data_ptr(), prob_nl.shape)` would be safer.
- `OrdinalSustain._optimise_parameters` (`OrdinalSustain.py:216-243`)
  contains shape-mismatch repair logic that pads/truncates likelihood
  outputs. This is a symptom of an earlier bug that should have been
  fixed at the source — paper over the cause rather than the symptom.
- `MixtureSustain` has no Torch variant. Same pattern as
  `TorchOrdinalSustain` would apply trivially.

## 5. Related work worth citing in the thesis

- **fastSuStaIn** (Lowther et al., the project this repo's GPU layer is
  modelled on). Same pattern: override `_calculate_likelihood_stage`,
  keep everything else CPU.
- **s-SuStaIn** (Tandon et al., MLR 2024,
  https://proceedings.mlr.press/v248/tandon24a.html). Clusters
  biomarkers simultaneously with subjects, reports an order-of-magnitude
  speedup. Orthogonal to GPU acceleration — could be stacked.
- **Sountsov, Carroll, Hoffman 2024** (arXiv:2411.04260). The reference
  point for §4.2.
- **Margossian et al. 2024** (Nested Rhat, arXiv:2110.13017). The
  convergence diagnostic that makes the "many short chains" regime
  practical.
- **Efficiently Vectorised MCMC on Modern Accelerators**
  (arXiv:2503.17405). Direct follow-up to Sountsov et al. with
  PyTorch-specific recommendations.

## 6. Summary

The current v2 acceleration is correct and well-engineered for what it
does. It is also a textbook example of the pattern Sountsov et al.
explicitly classify as suboptimal: a single chain whose inner loop
forces a host↔device round-trip per iteration. The clean next steps
are, in order:

1. Fix the float32 log guard (§4.4) — small but real correctness issue.
2. Move the inner MCMC loop on device, returning only the scalar
   log-likelihood per iteration (§4.1).
3. Vectorise across chains (§4.2), the Sountsov-style speedup that
   leverages SuStaIn's specific structure (small N, large M, shared
   data).
4. Optionally: incremental likelihood updates (§4.3) for an extra
   constant-factor win that is independent of hardware.
5. Drop or rewrite `parallel_mcmc.py` (§4.5) and `numpyro_sustain.py`
   (§4.6) — both are misleading in their current state.

A reasonable thesis-scope subset: §4.1 + §4.2 with a single
well-validated Torch path, drop the v1 batched-proposals code and the
broken `parallel_mcmc.py`, document why NUTS/HMC isn't applicable.
That delivers a defensible single-GPU implementation that follows the
paper's recommendation, and gives a clean comparison point (GPU
single-chain v2 vs GPU multi-chain v3) for the experimental section.
