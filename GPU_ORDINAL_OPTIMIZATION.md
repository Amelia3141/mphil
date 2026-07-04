# GPU-Accelerated OrdinalSustain Implementation

This document describes the GPU optimization of the OrdinalSustain algorithm using
PyTorch (`pySuStaIn/TorchOrdinalSustain.py` + `pySuStaIn/torch_likelihood.py`).

> **2026-07 correction.** An earlier version of this file described a naive
> *per-stage GPU loop* and quoted a 10–20× speedup copied from the ZScore
> benchmarks. Both were inaccurate. The code has since been rewritten to a single
> **batched gather** kernel (no per-stage loop), and the speedup claims here have
> been replaced with what the implementation actually guarantees plus honest,
> data-size-dependent expectations that must be measured on real CUDA hardware.

## Overview

`TorchOrdinalSustain` is a drop-in subclass of `OrdinalSustain`. It overrides
**only** the likelihood computation (`_calculate_likelihood_stage` and
`_calculate_likelihood`) and dispatches those to the GPU. Everything else — EM,
MCMC, `_optimise_parameters`, RNG, accept/reject — is inherited unchanged. This is
what guarantees the GPU run reproduces the CPU result rather than being a different
algorithm.

## How the speedup is achieved (and why it is bounded)

### The ordinal likelihood has a sequential dependency — but it is index-only

Which biomarkers are "abnormal" at stage *j* depends on stages 0…*j-1*. That
dependency is **pure index logic that never touches the data**. The implementation
exploits this:

1. **(CPU, microseconds)** Walk the sequence `S` once to build a padded index
   matrix: for each of the `N+1` stages, which columns of a combined
   `[log_prob_score | log_prob_nl]` tensor should be summed.
2. **(GPU, one batched op)** Gather all those columns for all stages at once via
   advanced indexing, mask out padding, sum across columns, and `exp`. That is
   **~4 GPU kernels per call, independent of `N`** — not `N` small kernels.

The expensive part — `log(prob_score)` / `log(prob_nl)` and their concatenation
into an `(M, N+B)` tensor — is **cached once per `sustainData` object** and reused
across every MCMC iteration (`get_log_combined_torch()` in
`torch_data_classes.py`).

### Why the old per-stage loop was removed

A straightforward port keeps the `for j in range(N)` loop on the GPU, launching one
small kernel per stage (~38 stages for 19 biomarkers × 2 levels). Per the kernel's
own docstring, that version ran **~11× slower than CPU** — the per-stage kernel
launch overhead dwarfs the tiny per-stage arithmetic. The batched-gather rewrite
gives the GPU enough work per call to overcome launch overhead, which is the whole
point of the current design.

### What limits the end-to-end speedup

- **Amdahl's law.** Only the likelihood runs on GPU. MCMC proposal generation,
  accept/reject, EM convergence, and RNG stay on CPU by design (that is what keeps
  results identical). The end-to-end speedup is therefore capped by the fraction of
  runtime spent in the likelihood — even an infinitely fast GPU likelihood yields a
  finite overall speedup.
- **Residual per-call host work.** The gather-index matrix is rebuilt in NumPy each
  call and a small int tensor is copied host→device. This cannot be cached because
  `S` changes every MCMC step. At small subject counts this overhead can erase the
  gain; at large `M` the GPU gather over all subjects dominates and amortises it.
- **Sweet spot = large M.** The design wins most at DICE scale (~10k subjects),
  where the per-subject parallelism is large relative to the fixed per-call cost.

## Numerical correctness (verified)

`use_gpu=True, force_float64=True` runs the identical algorithm at float64 for
exact CPU equivalence; `force_float64=False` uses float32 for production speed.
`use_gpu=True` engages CUDA **only when `torch.cuda.is_available()`** — otherwise it
falls back to CPU/float64 with a warning (it does not use Apple MPS).

Verified locally (CPU tensors, since no CUDA was available on the dev machine — the
gather/sum/exp kernel runs the same code on CPU and CUDA tensors):

| Check | Result |
|-------|--------|
| Batched-gather kernel vs CPU `_calculate_likelihood_stage` (20 sequences, n=800×19) | **max abs diff 8.7e-19** (identical) |
| `_calculate_likelihood_stage` full validation (10 sequences) | 0.00e+00 |
| `_calculate_likelihood` mixture, N_S=1 and N_S=2 | 0.00e+00 |
| Data-subset handling (reindex) | 0.00e+00 |
| Full `run_sustain_algorithm` pipeline | 100% stage match, corr 1.0 |

Run it: `python benchmark_ordinal_gpu.py --validate-only`.

## Performance expectations

**There is no verified speedup table yet** — the numbers must be measured on the
target GPU. The earlier "10–20×" figures were carried over from the ZScore model,
which is more vectorizable than the ordinal model and is **not** a valid proxy.

Realistic guidance for the ordinal path:

- Expect a **single-digit end-to-end multiple** (Amdahl-bounded), best at large `M`.
- Very small datasets may see little or no gain, or a slowdown, due to per-call host
  overhead.
- Measure before committing to a full run:

```bash
python benchmark_ordinal_gpu.py            # validation + CPU-vs-GPU table up to DICE scale
```

The benchmark prints CPU vs GPU timings for `_calculate_likelihood_stage` and for
the full pipeline at 100 → 15000 subjects, which is the real basis for planning.

## Usage Example

```python
from pySuStaIn import TorchOrdinalSustain

ordinal_sustain = TorchOrdinalSustain(
    prob_nl=prob_nl,           # (M, B) normal probabilities
    prob_score=prob_score,     # (M, B, num_scores) score probabilities
    score_vals=score_vals,     # (B, num_scores) score value matrix
    biomarker_labels=labels,
    N_startpoints=25,
    N_S_max=3,
    N_iterations_MCMC=100000,
    output_folder="./output",
    dataset_name="my_data",
    use_parallel_startpoints=False,   # single-process path for the GPU run
    seed=42,
    use_gpu=True,              # engages CUDA if available, else CPU fallback
    force_float64=False,       # float32 = speed; True = exact CPU-equivalent validation
)

results = ordinal_sustain.run_sustain_algorithm()
```

## Architecture

```
TorchOrdinalSustain  (subclass of OrdinalSustain)
  • inherits EM / MCMC / staging unchanged  -> identical results
  • overrides _calculate_likelihood_stage / _calculate_likelihood -> GPU dispatch
  • OOM fallback to the CPU superclass method
        │
        ├── TorchSustainBackend / DeviceManager   (CPU-or-CUDA, dtype, memory)
        ├── TorchOrdinalSustainData               (prob_nl, prob_score, cached log_combined)
        └── TorchOrdinalLikelihoodCalculator
                _calculate_likelihood_stage_torch():
                  CPU: build padded gather-index matrix by walking S
                  GPU: gather -> mask -> sum -> exp   (~4 kernels, no per-stage loop)
```

## Files

- `pySuStaIn/torch_likelihood.py` — `TorchOrdinalLikelihoodCalculator` (the batched
  gather kernel) + `create_ordinal_likelihood_calculator()`
- `pySuStaIn/TorchOrdinalSustain.py` — the drop-in subclass and GPU dispatch
- `pySuStaIn/torch_data_classes.py` — `TorchOrdinalSustainData`, `get_log_combined_torch()` cache
- `pySuStaIn/torch_backend.py` — device/precision management, `force_float64`
- `benchmark_ordinal_gpu.py` — validation suite + CPU-vs-GPU benchmarks

## Future improvements

1. Keep the gather-index construction on-device to remove the per-call host→device
   transfer (the main residual overhead).
2. Batch across subtypes (sequences) so N_S>1 dispatches in one kernel.
3. Mixed precision (float16) for very large `M`.

## References

- Original SuStaIn paper: https://doi.org/10.1038/s41467-018-05892-0
- Ordinal SuStaIn paper: https://doi.org/10.3389/frai.2021.613261
- PyTorch documentation: https://pytorch.org/docs/
