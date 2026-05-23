# Parallel MCMC chains for SuStaIn

This implementation runs independent MCMC chains in separate OS processes
via `multiprocessing`. It exists in two flavours:

1. **Standalone** — `pySuStaIn.parallel_mcmc.run_parallel_chains`. Works
   with any `AbstractSustain` subclass; the user passes the class and
   constructor kwargs, the function builds one instance per worker.
2. **Wrapper** — `ParallelTorchZScoreSustainMissingData`. Drops in for
   `TorchZScoreSustainMissingData`; the multi-chain MCMC kicks in
   inside `_estimate_uncertainty_sustain_model`.

## Why process-based rather than thread-based

SuStaIn's likelihood evaluation is pure NumPy/Python with no GIL
release. Threads share the GIL, so `n_threads` threads of NumPy code
run roughly as fast as 1 — see the `concurrent.futures` docs on the
GIL. The previous thread-based implementation in this repository
silently delivered no parallelism while reporting a "speedup" derived
from a load-imbalance ratio rather than an actual time comparison.

The current implementation uses `multiprocessing.ProcessPoolExecutor`
with the `spawn` start method (so it works on macOS and Windows and
does not inherit unwanted parent state like CUDA contexts).

## Caveats

- Each worker constructs a fresh SuStaIn instance, including
  re-running the `__init__` preprocessing. For typical input sizes
  this is negligible compared with MCMC.
- When `use_gpu=True`, each worker process allocates its own GPU
  context and its own copy of the data tensors. Running `n_chains > 2`
  on a single GPU may exhaust memory. For chain-level parallelism on
  one GPU the better pattern is to vectorise the chains as a leading
  tensor axis — see `ACCELERATION_METHOD_REVIEW.md` §4.2.
- The proposal scales (`seq_sigma`, `f_sigma`) are tuned once in the
  parent process and passed to all chains so they explore at the same
  scale. Each chain still draws independent proposals.
- Chains are pooled by concatenation along the iteration axis. They
  remain independent chains for diagnostic purposes (Rhat, ESS) — only
  the final ML point estimate is shared.

## Example

```python
from pySuStaIn.OrdinalSustain import OrdinalSustain
from pySuStaIn.parallel_mcmc import run_parallel_chains, combine_chain_samples

init_kwargs = dict(
    prob_nl=prob_nl, prob_score=prob_score, score_vals=score_vals,
    biomarker_labels=labels,
    N_startpoints=10, N_S_max=2, N_iterations_MCMC=10_000,
    output_folder="./out", dataset_name="test",
    use_parallel_startpoints=False, seed=42,
)

# seq_init / f_init typically come from EM
result = run_parallel_chains(
    sustain_class=OrdinalSustain,
    init_kwargs=init_kwargs,
    seq_init=seq_init, f_init=f_init,
    n_iterations=10_000,
    seq_sigma=1.0, f_sigma=0.01,
    n_chains=4,
)
print(f"wall {result['wall_time']:.1f}s, speedup {result['speedup']:.2f}x")

pooled = combine_chain_samples(result)
```

## What was removed

The previous version of this README claimed a 2-4x speedup from
`ThreadPoolExecutor`. That number was not achievable with the previous
implementation (GIL contention) and the benchmark code reporting it
returned mock data. The replacement is honest about the constraints.
See `ACCELERATION_METHOD_REVIEW.md` for the full discussion.
