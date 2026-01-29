# Instructions: Generating Synthetic Data for Ordinal Sustain

## Part 1: Rebuild the Synthetic Data Generator

### 1.1 Generation of cross-sectional data

Generate cross-sectional data that mimics DICE Registry structure:

1. **Match DICE variable names and formats exactly** (see Section 2 below)
2. **Generate realistic marginal distributions** using published prevalence rates
3. **Model realistic correlation structure** between variables (e.g., fatigue-pain correlation)
4. **Add appropriate missingness patterns** (not just MCAR; include MAR based on clinical logic)
5. **Let SuStaIn find whatever structure exists** in this realistic but not pre-structured data

### 1.2 DICE Registry Variable Mapping

Based on DICE_DataGuide.pdf, here are the relevant sections and variables you need to implement. Note: DICE uses categorical/binary responses, not 0-3 severity scales. You'll need to either:
- (A) Generate DICE-format data then convert to ordinal scores for SuStaIn, or
- (B) Generate ordinal scores directly but ensure marginal prevalences match DICE

**Demographics (Tier 1):**
- Age range
- Country
- Province/State
- Race
- Employment status
- Education level
- Sex at birth
- Gender
- Sexual orientation

**EDS/HSD Diagnosis (Tier 1 and 2):**
- Diagnosis status (diagnosed, suspected, etc.)
- Diagnosis type (hEDS, HSD, cEDS, vEDS, etc.)
- Year of diagnosis (Tier 2 only, or ranges for Tier 1)
- Genetic testing confirmation status
- Genetic testing results (Tier 2)

**General Health:**
- Fatigue (presence and likely severity implied)
- Sleep issues
- Weight fluctuation
- Rare-type signals

**Pain Symptoms:**
- Frequency of pain
- Pain interference with daily activities
- Pain medications and interventions
- Use of dietary changes for pain reduction
- Use of physical/manual therapies
- Use of cognitive/mindful therapies
- Supports and assistive devices

**Joints, Muscles, and Bones:**
- Scoliosis diagnosis
- Joint pain frequency and location
- Muscle pain frequency
- Muscle weakness
- Hypermobility and instability in joints
- Frequency and location of dislocated joints
- Frequency and location of joint subluxations

**Gynecological:**
- Pregnancy-related questions
- Uterine prolapse history
- Uterine rupture history
- PCOS diagnosis

**Urinary:**
- Presence of urinary symptoms (urgency, incontinence, etc.)

**Gastrointestinal:**
- Presence of GI symptoms (abdominal pain, bloating, etc.)
- Frequency of GI symptoms
- Bowel perforation history
- Abdominal hernias

**Allergy, Autoimmune, and Endocrine:**
- Autoimmune diagnoses
- Severe infection frequency
- Hormone-related symptoms
- Allergy diagnoses (asthma, allergic rhinitis)
- Allergy symptoms (anaphylaxis, angioedema, rash, etc.)
- **MCAS diagnosis** (critical for your triad)

**Mental Health:**
- Mental illness diagnoses (anxiety, depression, bipolar, etc.)
- Neurodevelopmental disorders (ADHD, ASD, etc.)
- Disability and impairments

**Neurological & Spinal:**
- Presence of neurological/spinal conditions (AAI, carpal tunnel, Chiari, etc.)
- Presence of neurological symptoms (confusion, fainting, headache, etc.)

**Cardiovascular:**
- Cardiovascular conditions diagnosis (arrhythmia, chest pain, **POTS**, etc.)
- Blood vessel conditions
- Carotid-cavernous sinus fistula
- Aortic root enlargement/aneurysm
- Mitral valve prolapse

**Head, Ears, Eyes, Nose, and Teeth:**
- Vision issues
- Hearing loss
- Tinnitus
- Oral features and symptoms

**Respiratory:**
- Chronic cough
- Difficulty breathing
- Shortness of breath
- Wheezing

**Hair and Skin:**
- Easy bruising
- Skin tears
- Hives
- Itching
- Poor wound healing
- Rashes
- Wide/sunken scars
- Stretch marks
- Soft/velvety skin
- Stretchy skin

**Blood and Lymphatic:**
- Prolonged bleeding
- Frequent nosebleeds
- Swollen lymph nodes

### 1.3 Prevalence Rates to Use

Use these from literature (cite Hakim et al. 2017, Tinkle et al. 2017, DICE publications):

| Symptom/Condition | Approximate Prevalence in EDS Population |
|-------------------|------------------------------------------|
| Chronic fatigue | 70-85% |
| Chronic pain | 75-90% |
| GI symptoms | 70-80% |
| Anxiety | 60-70% |
| Depression | 40-50% |
| POTS/dysautonomia | 40-50% in hEDS |
| MCAS (diagnosed) | 15-30% (highly variable) |
| Joint hypermobility (by definition in hEDS) | ~100% |
| Recurrent subluxations | 60-80% |
| Headaches/migraines | 50-70% |
| Urinary symptoms | 30-50% |
| Skin fragility/easy bruising | 60-80% |

### 1.4 Correlation Structure

Model these known clinical associations:

**Strong positive correlations (r ≈ 0.4-0.7):**
- Fatigue ↔ Pain
- Pain ↔ Depression
- Anxiety ↔ Depression
- GI symptoms ↔ MCAS symptoms
- Dysautonomia ↔ Fatigue
- Joint pain ↔ Subluxation frequency

**Moderate correlations (r ≈ 0.2-0.4):**
- Anxiety ↔ Dysautonomia (bidirectional in literature)
- MCAS ↔ Skin symptoms
- Pain ↔ Sleep problems
- Fatigue ↔ Cognitive symptoms

**Implementation approach:**
Use a multivariate normal distribution with a specified correlation matrix, then convert to ordinal via thresholds. Alternatively, use copulas to model dependencies while maintaining correct marginals.

### 1.5 Missingness Patterns

DICE registry missingness is not purely MCAR. Model:

- **MAR on demographics:** Gynae questions missing for males
- **MAR on severity:** People with no symptoms may skip detailed follow-ups
- **MCAR noise:** ~5-10% random missingness across all variables
- **Section-based missingness:** If someone doesn't complete a section, all variables in that section are missing

### 1.6 Output Format

The synthetic data output should:
- Use DICE-like column names (or document the mapping clearly)
- Include only variables that would appear in a real DICE data extract
- NOT include `subtype_true` or `stage_true` columns
- Document what prevalence rates and correlation structure were used (for reproducibility)
- Be in a format that your SuStaIn preprocessing pipeline can accept

---

## Part 2: Fix SuStaIn Validation Issues

### 2.1 Parallel Processing Artefacts

**Problem:** `run_validation.py` uses `use_parallel_startpoints=True`. Parallel aggregation could introduce bias if results are averaged rather than selected/counted.

**Action required:**
1. Run SuStaIn with `use_parallel_startpoints=False`
2. Compare results to the parallel run
3. If results differ substantially, investigate pySuStaIn's parallelisation implementation
4. Document whether parallel vs sequential gives equivalent results

**Specific test:**
```python
# Sequential run
sustain_seq = OrdinalSustain(..., use_parallel_startpoints=False, seed=42)
results_seq = sustain_seq.run_sustain_algorithm()

# Parallel run
sustain_par = OrdinalSustain(..., use_parallel_startpoints=True, seed=42)
results_par = sustain_par.run_sustain_algorithm()

# Compare: Are ml_subtype, ml_stage, samples_sequence equivalent?
```

### 2.2 MCMC Convergence

**Problem:** 10,000 iterations with 10 startpoints. Mean stage probability of 0.723 (vs 0.951 for subtypes) suggests uncertainty in stage assignment.

**Actions required:**
1. **Check trace plots:** Does the log-likelihood stabilise, or is it still trending?
2. **Check Gelman-Rubin R-hat:** Compare chains from different startpoints (should be < 1.1)
3. **Increase iterations if needed:** Try 25,000 or 50,000 iterations
4. **Effective sample size:** Calculate ESS; if low, chains haven't mixed well

**Implementation:**
```python
# Save samples_sequence for each startpoint
# Plot log-likelihood trace
# Calculate R-hat across chains

# If not converged:
sustain = OrdinalSustain(..., N_iterations_MCMC=50000, ...)
```

### 2.3 Spurious Subtype 3

**Problem:** 11 patients (2.2%) assigned to non-existent subtype 3 (patients 34, 115, 117, 455, 461 and 6 others). Data was generated with 2 subtypes but SuStaIn found a third.

**This is exactly the artefact Wati warned about:** algorithms can always create clusters even when they don't represent genuine biological structure.

**Actions required:**

1. **Check model selection:** What does the CVIC (cross-validation information criterion) or BIC say about 2 vs 3 subtypes?
   ```python
   # Look at samples_f for each number of subtypes
   # Compare log-likelihood penalised for complexity
   ```

2. **Examine the "subtype 3" patients:** What makes them different?
   ```python
   results_df = pd.read_csv('validation_results.csv')
   subtype3 = results_df[results_df['inferred_subtype'] == 3]
   # Look at their biomarker profiles
   # Are they outliers? Early/late stage? Unusual combinations?
   ```

3. **Bootstrap stability analysis:** Does subtype 3 appear consistently across bootstrap samples, or is it unstable?
   ```python
   # Run SuStaIn on 100 bootstrap samples
   # Count how often each patient is assigned to subtype 3
   # If assignment is inconsistent, the cluster is spurious
   ```

4. **Force 2 subtypes:** Run with `N_S_max=2` and compare fit
   ```python
   sustain_2 = OrdinalSustain(..., N_S_max=2, ...)
   ```

### 2.4 What This Means for Real Data

If SuStaIn creates spurious clusters even on clean synthetic data with known structure, you need a validation framework for real DICE data that can distinguish genuine subtypes from algorithmic artefacts:

1. **Cox regression validation:** Do identified subtypes predict differential clinical outcomes (hospitalisations, surgeries, disability progression)?
2. **Replication:** Do subtypes replicate in held-out data or external cohorts?
3. **Clinical coherence:** Do subtypes correspond to known clinical phenotypes or suggest plausible pathophysiology?
4. **Stability:** Are subtypes stable under bootstrap resampling?

---

## Part 3: Implementation Order

### Phase 1: Fix Synthetic Data (Priority: HIGH)

1. Read DICE_DataGuide.pdf in detail
2. Create variable mapping document (DICE name → your variable name)
3. Research prevalence rates from literature (cite sources)
4. Design correlation matrix based on clinical associations
5. Implement new generator WITHOUT progression sequences
6. Validate marginal distributions match targets
7. Validate correlation structure approximately matches targets
8. Document assumptions and limitations

### Phase 2: Diagnose Current SuStaIn Issues (Priority: MEDIUM)

1. Run parallel vs sequential comparison
2. Run extended MCMC (50k iterations)
3. Examine subtype 3 patients
4. Run bootstrap stability analysis
5. Document findings

### Phase 3: Run SuStaIn on New Synthetic Data (Priority: HIGH)

1. Apply SuStaIn to DICE-like synthetic data (no pre-defined subtypes)
2. Whatever subtypes emerge, evaluate them for:
   - Stability (bootstrap)
   - Clinical coherence (do biomarker profiles make sense?)
   - Separation quality (silhouette, gap statistic)
3. Document: "SuStaIn found N subtypes in realistic synthetic data. Here's what they look like."

### Phase 4: Prepare Validation Framework for Real Data (Priority: MEDIUM)

1. Pre-specify primary analysis plan
2. Define what would constitute "clinically meaningful" subtypes
3. Plan Cox regression validation
4. Plan sensitivity analyses (missingness, outliers, model assumptions)

---

## Appendix: Code Skeleton for New Generator

```python
"""
Synthetic Data Generator v2: DICE-like structure without pre-defined subtypes
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

def define_prevalence_rates():
    """Return dict of symptom prevalence rates from literature."""
    return {
        'fatigue': 0.78,
        'chronic_pain': 0.82,
        'gi_symptoms': 0.75,
        'anxiety': 0.65,
        'depression': 0.45,
        'pots_dysautonomia': 0.45,
        'mcas': 0.25,
        'joint_hypermobility': 0.98,
        'subluxations': 0.70,
        'headaches': 0.60,
        'urinary_symptoms': 0.40,
        'skin_fragility': 0.70,
        # ... add all DICE variables
    }

def define_correlation_matrix():
    """Return correlation matrix for multivariate generation."""
    # Define based on clinical associations
    # Use variable ordering consistent with prevalence dict
    n_vars = len(define_prevalence_rates())
    corr = np.eye(n_vars)

    # Set specific correlations
    # e.g., corr[fatigue_idx, pain_idx] = 0.5
    # ...

    return corr

def generate_latent_continuous(n_patients, prevalence, correlation, seed=42):
    """Generate latent continuous variables with specified correlation."""
    np.random.seed(seed)
    n_vars = len(prevalence)

    # Multivariate normal with correlation structure
    mean = np.zeros(n_vars)
    samples = multivariate_normal.rvs(mean=mean, cov=correlation, size=n_patients)

    return samples

def threshold_to_ordinal(continuous_values, prevalence, n_levels=4):
    """Convert continuous latent to ordinal scores matching prevalence."""
    # Threshold such that P(score > 0) = prevalence
    # Distribute remaining probability across levels 1, 2, 3
    # ...
    pass

def add_missingness(df, mcar_rate=0.05, mar_rules=None):
    """Add realistic missingness patterns."""
    # MCAR component
    # MAR component (e.g., gynae missing for males)
    # ...
    pass

def generate_dice_like_data(n_patients=5000, seed=42):
    """Main generation function."""
    prevalence = define_prevalence_rates()
    correlation = define_correlation_matrix()

    # Generate latent continuous
    latent = generate_latent_continuous(n_patients, prevalence, correlation, seed)

    # Convert to ordinal
    ordinal_data = {}
    for i, (var_name, prev) in enumerate(prevalence.items()):
        ordinal_data[var_name] = threshold_to_ordinal(latent[:, i], prev)

    df = pd.DataFrame(ordinal_data)

    # Add demographics
    df = add_demographics(df, n_patients, seed)

    # Add missingness
    df = add_missingness(df)

    return df
```

---

## References for Prevalence Rates

- Hakim A, et al. (2017). "Cardiovascular autonomic dysfunction in Ehlers-Danlos syndrome." Am J Med Genet C, 175:212-218.
- Tinkle B, et al. (2017). "Hypermobile EDS: Clinical description and natural history." Am J Med Genet C, 175:48-69.
- Halverson CME, et al. (2023). "Comorbidity, misdiagnoses, and the diagnostic odyssey in hEDS." Genetics in Medicine Open, 1:100812.
- The Ehlers-Danlos Society DICE Registry publications (check their website for latest reports)
