# Synthetic Data Generation Instructions

This document describes how the synthetic DICE-like dataset (`dice_synthetic_v2.csv`) was generated using an LLM-assisted approach.

## Purpose

The synthetic data generator creates **realistic EDS/HSD symptom data WITHOUT pre-encoded subtypes**. This is critical for validating SuStaIn methodology:

- **Original problem**: Previous generators pre-encoded subtypes then "discovered" them (circular validation)
- **Solution**: Generate data with realistic correlation structure but NO ground truth subtypes
- **Validation goal**: Pre-specified criteria should REJECT spurious patterns SuStaIn finds in null data

## Generation Methodology

### Step 1: Literature-Based Prevalence Rates

Symptom prevalence rates were extracted from published EDS/HSD literature:

| Symptom | Prevalence | Source |
|---------|------------|--------|
| Joint hypermobility | 98% | By definition (hEDS criteria) |
| Joint pain | 85% | Tinkle et al. 2017 |
| Chronic pain | 82% | Hakim et al. 2017 |
| Fatigue | 78% | Halverson et al. 2023 |
| GI symptoms | 75% | Multiple sources |
| Subluxations | 70% | Tinkle et al. 2017 |
| Skin fragility | 70% | Tinkle et al. 2017 |
| Sleep disturbance | 70% | DICE Registry data |
| Anxiety | 65% | Halverson et al. 2023 |
| Headaches/migraines | 60% | Multiple sources |
| Cognitive fog | 60% | Patient reports |
| Allergies | 55% | Common comorbidity |
| Dysautonomia/POTS | 45% | Roma et al. 2018 |
| Depression | 45% | Halverson et al. 2023 |
| TMD/jaw | 45% | Clinical observation |
| Vision issues | 40% | Various |
| Urinary symptoms | 40% | Various |
| Hearing/tinnitus | 35% | Various |
| MCAS | 25% | Highly variable estimates |

### Step 2: Clinically-Plausible Correlation Structure

Correlations between symptoms were defined based on known clinical associations:

**Strong correlations (r = 0.45-0.60):**
- Fatigue ↔ Chronic pain (shared pathophysiology)
- Pain ↔ Depression (bidirectional relationship)
- Anxiety ↔ Depression (high comorbidity)
- Joint pain ↔ Subluxations (mechanical)
- Fatigue ↔ Sleep disturbance
- GI symptoms ↔ MCAS (mast cell involvement)

**Moderate correlations (r = 0.25-0.40):**
- Anxiety ↔ Dysautonomia
- MCAS ↔ Skin symptoms
- Pain ↔ Sleep problems
- Fatigue ↔ Cognitive fog
- Headaches ↔ Dysautonomia

**Weak correlations (r = 0.15-0.25):**
- Cross-domain links representing general illness burden
- Ensures network connectivity without separable clusters

### Step 3: Latent Variable Generation

1. **Multivariate normal sampling**: Generate continuous latent variables with the specified correlation matrix
2. **Ordinal thresholding**: Convert continuous values to 4-level ordinal scores (0-3)
   - Threshold set so P(score > 0) = target prevalence
   - Severity gradient: 50% mild, 35% moderate, 15% severe

### Step 4: Demographics

Generated to match DICE Registry patterns:
- **Age**: Mean 36, SD 12, range 18-80
- **Sex**: 90% female
- **Diagnosis type**: 75% hEDS, 15% HSD, 7% cEDS, 2% vEDS, 1% other

### Step 5: Missingness Patterns

- **MCAR**: 5% random missingness across all variables
- **MAR**: Additional 3% missingness for those with score = 0 (skip patterns)

## Code Implementation

The generator is implemented in `generate_dice_like_data.py`:

```bash
# Generate 5000 patients with seed 42
python generate_dice_like_data.py --n_patients 5000 --seed 42 --output dice_synthetic_v2.csv

# With validation output
python generate_dice_like_data.py -n 5000 -s 42 -o dice_synthetic_v2.csv --validate
```

### Key Functions

1. `get_prevalence_rates()` - Returns literature-based prevalence
2. `get_correlation_matrix()` - Defines symptom correlations
3. `nearest_positive_definite()` - Ensures valid correlation matrix
4. `generate_latent_continuous()` - Samples from multivariate normal
5. `threshold_to_ordinal()` - Converts to ordinal with severity gradient
6. `add_demographics()` - Generates demographic variables
7. `add_missingness()` - Applies MCAR and MAR patterns

## Validation Results

The generated data was validated against targets:

### Correlation Fidelity
- Target-Observed correlation: **r = 0.991**
- Maximum pairwise correlation: **0.581** (fatigue ↔ chronic pain)
- All correlations < 0.8 threshold

### No Hidden Structure
GMM comparison (k = 1-5 components):
- Silhouette scores: 0.08-0.10 (indicating NO cluster structure)
- BIC improvement trivial relative to silhouette
- **Conclusion**: No hidden subtypes in data

### SuStaIn Results on Null Data
- SuStaIn identifies 2 "subtypes" (as expected - it always finds patterns)
- **Criterion 2 FAILS**: Max Cohen's d = 0.27 (need d ≥ 0.5 on ≥ 3 domains)
- Pre-specified criteria correctly identify these as artefacts

## File Contents

**dice_synthetic_v2.csv** (5000 rows, 25 columns):

| Column | Type | Description |
|--------|------|-------------|
| patient_id | int | Unique identifier |
| age | int | Patient age (18-80) |
| sex | str | F/M |
| diagnosis_type | str | hEDS/HSD/cEDS/vEDS/other |
| year_of_diagnosis | int | 2000-2024 |
| years_since_diagnosis | int | Derived |
| fatigue | Int64 | 0-3 ordinal |
| chronic_pain | Int64 | 0-3 ordinal |
| joint_hypermobility | Int64 | 0-3 ordinal |
| joint_pain | Int64 | 0-3 ordinal |
| subluxations | Int64 | 0-3 ordinal |
| gi_symptoms | Int64 | 0-3 ordinal |
| dysautonomia_pots | Int64 | 0-3 ordinal |
| anxiety | Int64 | 0-3 ordinal |
| depression | Int64 | 0-3 ordinal |
| mcas | Int64 | 0-3 ordinal |
| allergies | Int64 | 0-3 ordinal |
| headaches_migraines | Int64 | 0-3 ordinal |
| skin_fragility | Int64 | 0-3 ordinal |
| urinary_symptoms | Int64 | 0-3 ordinal |
| sleep_disturbance | Int64 | 0-3 ordinal |
| tmd_jaw | Int64 | 0-3 ordinal |
| vision_issues | Int64 | 0-3 ordinal |
| hearing_tinnitus | Int64 | 0-3 ordinal |
| cognitive_fog | Int64 | 0-3 ordinal |

## References

1. Hakim A, et al. (2017). Cardiovascular autonomic dysfunction in Ehlers-Danlos syndrome. Am J Med Genet C, 175:212-218.

2. Tinkle B, et al. (2017). Hypermobile Ehlers-Danlos syndrome: Clinical description and natural history. Am J Med Genet C, 175:48-69.

3. Halverson CME, et al. (2023). Comorbidity, healthcare utilization, and diagnostic odyssey in hypermobile Ehlers-Danlos syndrome. Genetics in Medicine Open, 1:100812.

4. Roma M, et al. (2018). Postural tachycardia syndrome and other forms of orthostatic intolerance in Ehlers-Danlos syndrome. Autonomic Neuroscience, 215:89-96.

5. Young AL, et al. (2018). Uncovering the heterogeneity and temporal complexity of neurodegenerative diseases with Subtype and Stage Inference. Nature Communications, 9:4273.

## Reproducibility

To regenerate the exact dataset:

```bash
cd /path/to/mphil_repo
python generate_dice_like_data.py -n 5000 -s 42 -o data/dice_synthetic_v2.csv
```

The random seed (42) ensures deterministic output.

## Important Notes

1. **NO GROUND TRUTH**: This data has no true subtypes or stages
2. **VALIDATION ONLY**: Use this data to test that pre-specified criteria work
3. **NOT FOR TRAINING**: Do not use this data to "train" or "tune" SuStaIn parameters
4. **REAL DATA REQUIRED**: Clinical conclusions require real DICE Registry data
