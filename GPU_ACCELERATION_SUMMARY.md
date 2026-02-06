# GPU Acceleration of Ordinal SuStaIn — Summary for Presentation

## Slide: Initial Approach (v1) — What We Had

- Overrode the **entire MCMC loop** to batch 128 proposals on GPU at once
- Generated all proposals from the **same state**, then accepted/rejected serially
- Pre-loaded all data to GPU at `__init__` time, **ignored the `sustainData` argument** in likelihood calls
- Used separate `_single_sequence_likelihood()`, `_batch_loglike()`, `_single_mixture_loglike()` methods
- ~330 lines of custom GPU code replacing core algorithm logic

### Problems with v1
- **Different Markov chain**: Batching 128 proposals from the same state is a fundamentally different MCMC scheme — not just a faster version of the same algorithm
- **Confounding factor**: Cannot tell if result differences come from GPU acceleration or from the changed algorithm
- **Data subset bug**: Always used full dataset even when `AbstractSustain` passed a subset (breaks cross-validation, multi-subtype models with N_S >= 2)
- **Float32 on GPU vs float64 on CPU**: Additional source of numerical divergence with no option to control it
- **f perturbation difference**: For N_S > 1, CPU and GPU generated fraction perturbations differently (scalar vs vector `standard_normal`)

---

## Slide: The fastSuStaIn Approach — What Actually Works

- An existing GPU-accelerated SuStaIn implementation (Lowther et al.) for Z-score models
- Key insight: **only override the likelihood computation, leave everything else untouched**
- The MCMC loop, EM, `_optimise_parameters`, `_perform_mcmc` all run identically to CPU
- Each `_calculate_likelihood_stage()` call dispatches to GPU — same algorithm, faster math
- Data transferred to GPU **once** and cached; only the tiny sequence S (~22 elements) crosses CPU-GPU boundary per call

### Why it works
- `_calculate_likelihood_stage()` is called **hundreds of thousands of times** during a SuStaIn run
- Each call does `torch.prod()` across all M subjects on GPU — massively parallel
- The sequential loop over N stages (~22 iterations) stays as a Python loop, but within each iteration the subject-level computation is vectorized across all M subjects on GPU
- No algorithm logic changes = **no confounding factors**

---

## Slide: What We Changed (v2)

### Rewrote `TorchOrdinalSustain.py` (330 lines → 100 lines)
- **Removed**: `_perform_mcmc()` override, `_batch_loglike()`, `_single_mixture_loglike()`, `_single_sequence_likelihood()`, pre-loaded GPU tensors, fragile `sustainData` attribute search
- **Added**: `_ensure_torch_data(sustainData)` — wraps any `OrdinalSustainData` as a `TorchOrdinalSustainData` on-the-fly with caching, **respecting the actual `sustainData` argument** (fixes subset bug)
- **Override only**: `_calculate_likelihood_stage()` and `_calculate_likelihood()` to dispatch to GPU

### Added `force_float64` option to `torch_backend.py`
- Default: float32 on GPU (fast, ~2x memory bandwidth savings)
- Validation: float64 on GPU (exact CPU equivalence, no precision confounding)
- Allows running the **identical algorithm with identical precision** on GPU for validation, then switching to float32 for production speed

### Rewrote `benchmark_ordinal_gpu.py` — 4 validation tests
1. `_calculate_likelihood_stage`: GPU vs CPU on random sequences (10 tests, tolerance 1e-10)
2. `_calculate_likelihood`: Full mixture model, N_S=1 and N_S=2 (5 tests each)
3. **Data subset handling**: Verifies the old bug is fixed — GPU correctly computes on subsets
4. **Full pipeline**: `run_sustain_algorithm()` end-to-end, compares sequences and stages

---

## Slide: How It Speeds Things Up Without Affecting Results

### Guarantee of equivalence
- Same RNG sequence (seed unchanged, no custom `_perform_mcmc`)
- Same algorithm flow (EM → MCMC → staging, all inherited from `OrdinalSustain`)
- Same numerical precision (float64 option on GPU)
- Same data handling (respects `sustainData` argument including subsets)
- **Verified**: 0.00e+00 maximum difference on all likelihood tests

### Where the speed comes from
- **Within `_calculate_likelihood_stage()`**: `torch.prod()` across M subjects runs on GPU
- **Data stays on GPU**: `prob_nl` (M × B) and `prob_score` (M × N) transferred once, cached
- **Per-call overhead is tiny**: Only sequence S (~22 floats) crosses CPU-GPU per iteration
- **Expected speedup**: 5-20x for likelihood calls depending on dataset size
- **Scales with data**: Larger M (more subjects) = better GPU utilization

### What stays on CPU
- MCMC proposal generation (constraint logic, RNG)
- Accept/reject decisions
- EM convergence checks
- All I/O and result storage

---

## Slide: Validation Results (Local, CPU-only — no CUDA on MacBook M2)

| Test | Result |
|------|--------|
| `_calculate_likelihood_stage` (10 random sequences) | **PASS** — 0.00e+00 max diff |
| `_calculate_likelihood` N_S=1 (5 tests) | **PASS** — 0.00e+00 max diff |
| `_calculate_likelihood` N_S=2 (5 tests) | **PASS** — 0.00e+00 max diff |
| Data subset handling (reindex bug fix) | **PASS** — correct shapes and values |

**Next step**: Run full benchmark on Google Colab (T4 GPU) or UCL cluster to measure actual GPU speedup.

---

## Slide: Next Steps for Benchmarking

1. **Run `benchmark_ordinal_gpu.py` on Colab with T4 GPU**
   - Validate with float64: confirm 0 diff on GPU hardware
   - Benchmark with float32: measure speedup at 100-5000 subjects

2. **Run `colab_gpu_benchmark.py` (full pipeline)**
   - CPU sequential vs CPU parallel vs GPU at 200-2000 subjects
   - Measure end-to-end speedup including MCMC

3. **Scale test with DICE-like data (n=5000, 19 biomarkers)**
   - This is the real target dataset size
   - Expected: significant speedup since GPU parallelism scales with n_subjects

4. **Document for thesis**
   - Speedup table at various dataset sizes
   - Validation evidence (0 diff with float64)
   - Comparison with fastSuStaIn Z-score results
