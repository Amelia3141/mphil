# Ordinal SuStaIn for EDS/HSD Subtyping

This repository contains code for applying Ordinal SuStaIn to EDS/HSD registry data, developed as part of an MPhil thesis at UCL Institute of Health Informatics.

## Paper Reference

*Phenotypic Subtypes in Ehlers-Danlos Syndrome: A Data-Driven Approach Using Ordinal SuStaIn*
MPhil Thesis, University College London (in progress)

Supervised by Dr Ken Li and Dr Watjana Lilaonitkul.

## Updates

**2026-01-27**: Added known-structure recovery test (Experiment 2) and parallel/sequential equivalence test (Experiment 4).

**2026-01-26**: Rebuilt synthetic data generator to remove circular validation. Previously, subtypes were pre-encoded then "discovered" by SuStaIn (circular). New generator creates DICE-like correlation structure WITHOUT ground truth subtypes.

**2026-01-26**: Added pre-specified clinical meaningfulness criteria with citations (Steinley 2004, Vinh 2010, Young 2018).

## Requirements

- Python >= 3.9
- pySuStaIn (from ucl-pond/pySuStaIn)
- numpy, pandas, scipy, scikit-learn

Install via:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate synthetic DICE-like data (NO ground truth subtypes):
```bash
python generate_dice_like_data.py --n_patients 5000 --seed 42 --output dice_synthetic.csv
```

### 2. Run SuStaIn analysis:
```bash
python run_sustain_dice_data.py --input dice_synthetic.csv --n_subtypes_max 4 \
    --n_startpoints 25 --n_iterations 50000 --output sustain_results/
```

### 3. Evaluate clinical meaningfulness:
```bash
python evaluate_sustain_results.py --data dice_synthetic.csv \
    --pickle_dir sustain_results/pickle_files --n_subtypes 2
```

### 4. Run bootstrap stability analysis:
```bash
python run_bootstrap_stability.py --input dice_synthetic.csv --n_bootstrap 100 \
    --output bootstrap_results/
```

## Validation Experiments

### Experiment 1: Null Data Test
Demonstrates that SuStaIn finds "subtypes" in data without ground truth, but pre-specified criteria correctly reject them as artefacts.

```bash
python generate_dice_like_data.py -n 5000 -o null_data.csv
python run_sustain_dice_data.py -i null_data.csv -o null_results/ --n_subtypes_max 3
python evaluate_sustain_results.py -d null_data.csv -p null_results/pickle_files -n 2
```

**Expected result**: Criterion 2 (Cohen's d >= 0.5 on >= 3 domains) FAILS.

### Experiment 2: Known Structure Recovery
Verifies SuStaIn can recover genuine subtypes when they exist.

```bash
python generate_known_subtypes_data.py -n 2000 -s 3 -o known_data.csv --ground_truth known_truth.csv
python run_known_subtypes_validation.py -d known_data.csv -g known_truth.csv -o known_results/
```

**Expected result**: ARI >= 0.8 between predicted and true subtypes.

### Experiment 3: Bootstrap Stability
Tests reproducibility across resampled data.

```bash
python run_bootstrap_stability.py -i dice_synthetic.csv -n 100 -o bootstrap_results/
```

**Expected result**: Mean ARI >= 0.6 for stable subtypes.

### Experiment 4: Parallel vs Sequential Equivalence
Verifies parallelization doesn't affect results.

```bash
python run_parallel_sequential_test.py -d dice_synthetic.csv -n 500 -o equiv_test/
```

**Expected result**: ARI > 0.99 between parallel and sequential runs.

## Results on Synthetic DICE-like Data

Results on n=5,000 synthetic patients with NO ground truth subtypes:

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| 1. Prevalence | >= 10% | 48.0% min | PASS |
| 2. Distinctiveness | d >= 0.5 on >= 3 domains | d = 0.27 max | FAIL |
| 3. Bootstrap stability | ARI >= 0.6 | PENDING | - |
| 4. Model selection | Best n by CVIC | PENDING | - |
| 5. Not severity-only | r < 0.7 with burden | r = 0.047 | PASS |
| 6. Plausible sequences | Manual review | See output | - |

**Interpretation**: SuStaIn identifies subtypes in null data, but pre-specified criteria correctly identify them as artefacts (failing distinctiveness criterion). This validates the approach.

## File Structure

```
.
├── generate_dice_like_data.py          # Synthetic data generator (NO subtypes)
├── generate_known_subtypes_data.py     # Validation data generator (WITH subtypes)
├── run_sustain_dice_data.py            # Main SuStaIn analysis
├── run_known_subtypes_validation.py    # Experiment 2: known structure recovery
├── run_bootstrap_stability.py          # Experiment 3: bootstrap stability
├── run_parallel_sequential_test.py     # Experiment 4: equivalence test
├── evaluate_sustain_results.py         # Clinical meaningfulness evaluation
├── validate_correlations.py            # Correlation matrix validation
│
├── DICE_VARIABLE_MAPPING.md            # Variable definitions with prevalence citations
├── CLINICAL_MEANINGFULNESS_CRITERIA.md # Pre-specified evaluation criteria
├── COX_REGRESSION_FRAMEWORK.md         # Outcome validation plan
├── VERIFICATION_SUMMARY.md             # All validation test results
│
├── data/
│   └── dice_synthetic_v2.csv           # Generated null data (n=5000)
│
├── pySuStaIn/                          # Modified pySuStaIn with MCMC determinism fix
└── tests/                              # Validation tests
```

## Key Design Decisions

### Why Pre-Specified Criteria?

SuStaIn will always find patterns (it's designed to). The scientific question is whether those patterns are clinically meaningful. Pre-specifying criteria BEFORE seeing results prevents:
- Post-hoc rationalization
- p-hacking via criterion selection
- Circular validation

### Why Correlation-Based Synthetic Data?

The DICE Registry has known symptom correlations (e.g., fatigue-pain r~0.5). Synthetic data preserves this structure without encoding subtypes, creating a realistic "null" dataset.

### Domain Definitions

Domains are based on standard EDS clinical categorization (Tinkle et al. 2017), NOT derived from SuStaIn results:

- **Musculoskeletal**: joint_pain, subluxations, joint_hypermobility
- **Autonomic**: dysautonomia_pots, gi_symptoms
- **Neurological**: headaches_migraines, cognitive_fog
- **Mental Health**: anxiety, depression
- **Constitutional**: fatigue, chronic_pain, sleep_disturbance

## Validation

See `VERIFICATION_SUMMARY.md` for complete validation results:
- No hidden subtype structure in generator (silhouette < 0.11)
- Correlation matrix validated (target-observed r = 0.991)
- Maximum correlation < 0.8 (actual max = 0.581)
- Parallel/sequential equivalence verified (MCMC determinism fix applied)

## Hardware and Timing

Development performed on MacBook Pro M2 (16GB RAM).

Typical run times for n=5,000 patients:
- 1 subtype model: ~30 minutes
- 2 subtype model: ~60 minutes
- 3 subtype model: ~90 minutes
- Full analysis (1-4 subtypes): ~4 hours

For production runs on real DICE data (n~45,000), UCL cluster access recommended.

## Citation

If you use this code, please cite:

```
@phdthesis{[author]2026,
  title={Phenotypic Subtypes in Ehlers-Danlos Syndrome: A Data-Driven Approach},
  author={[Author]},
  year={2026},
  school={University College London}
}
```

Also cite:
- Aksman LM, Wijeratne PA, et al. (2021). pySuStaIn: A Python implementation of the SuStaIn algorithm. SoftwareX.
- Young AL, et al. (2021). Ordinal SuStaIn: Subtype and Stage Inference for clinical scores, visual ratings, and other discrete data. Frontiers in Artificial Intelligence.

## Acknowledgments

- Supervised by Dr Ken Li and Dr Watjana Lilaonitkul, UCL Institute of Health Informatics
- Data access pending from The Ehlers-Danlos Society DICE Global Registry
- Built on pySuStaIn from UCL POND group
