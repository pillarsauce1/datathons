# Datathons

Notebooks and write-ups from two Kaggle-hosted datathons. Both are clinical survival-outcome problems on tabular data.

| Folder | Competition | Data | Task | Target |
|---|---|---|---|---|
| [`insight/`](insight/) | [Insight 2.0](https://www.kaggle.com/competitions/insight-2-0) | SEER lung cancer registry | Binary classification | `vital_status` (Dead / Alive) |
| [`bubt/`](bubt/) | [BAIC — BUBT Artificial Intelligence Conquest 2025](https://www.kaggle.com/competitions/bubt-artificial-intelligence-conquest-2025) | Synthetic cirrhosis cohort (from the Mayo Clinic PBC trial) | 3-class probability prediction | `Status` (C / CL / D) |

> **Note on naming:** "Cirrus" is the team name used for Insight 2.0. The `Cirrus_notebook` and `Cirrus.pdf` files are about **lung cancer (SEER)**, not cirrhosis. The cirrhosis work is in `bubt/`.

## Repository structure

```
datathons/
├── insight/
│   ├── Cirrus_notebook.ipynb   # preprocessing, feature engineering, GLM / Lasso / GBDT ensemble
│   └── Cirrus.pdf              # presentation slides
├── bubt/
│   └── bubt_finalnotebook.ipynb  # EDA, Kaplan–Meier, epidemiological tests, XGBoost submission
└── README.md
```

## Data

The competition data is **not included** in this repo. Download it from each competition's *Data* tab on Kaggle. You must be logged in and accept the competition rules(Private competitions). The notebooks expect the standard Kaggle paths (`/kaggle/input/<competition-slug>/...`).

---

## Insight 2.0 — SEER lung cancer survival

### Problem
Predict whether a lung cancer patient is dead or alive (`vital_status`) from 35 SEER registry variables. The variables fall into eight groups: demographics, diagnostic timeline, tumour anatomy and histology, tumour size, staging and extension, lymph node spread, metastasis sites at diagnosis, and treatment and surgery. The classes are imbalanced, at roughly 83% Dead.

### What makes this dataset tricky
Most of the work went into understanding *why* values are missing. The missingness is largely **structural**, not random:

- **Coding-system eras.** SEER changed coding schemes over the diagnosis years, so several variables are blank by design depending on the year:
  - `cs_tumor_size20042015` and `cs_extension20042015` (Collaborative Stage) are recorded only through 2015. `tumor_size_summary` takes over from 2016. Together they cover every record, and they were coalesced into one tumour-size feature, with `tumor_size_overtime` as the primary source.
  - `summary_stage` and `grade_recode_thru2017` end in 2017. The `derived_eod2018{t,n,m}_recode2018` TNM variables begin in 2018. Each era is blank in the other's columns.
- **Sentinel codes.** In the lymph node counts, values of 90 and above are category codes, not counts. For example, `98` in `regional_nodes_positive` means no nodes were examined. In tumour size, `999` means unknown. All of these were decoded before modelling.
- **"Unknown" is informative.** Patients coded *Unknown* on the four metastasis-site variables had **higher** mortality (≈98%) than those with confirmed metastasis (≈92–96%). Missing `rx_summ_scope_reglnsur2003` marks an inoperable, largely metastatic cohort (≈90% dead vs ≈70%). These values were kept as their own categories rather than imputed.
- **Diagnostic confirmation as a frailty proxy.** Patients diagnosed by imaging or clinical assessment only, without a biopsy, die at much higher rates, likely because they were too ill for a biopsy.

### Approach
1. **Preprocessing:** era-aware coalescing of tumour size and stage variables, sentinel-code decoding, and explicit "unknown" flags.
2. **Feature engineering:** 28 domain features, including ordinal grade, node positivity ratio, metastasis counts, surgery and radiation flags, and modern-EOD-era indicators. Also an out-of-fold empirical-Bayes mortality prior over 44 cohorts of (year × surgery × metastasis).
3. **Models compared:**
   - Classical logistic regression (GLM), on an isolated design matrix to avoid leakage
   - Lasso (L1) logistic regression, for variable selection
   - Tri-model GBDT ensemble (CatBoost + XGBoost + LightGBM, 5-fold stratified CV), blended with Nelder–Mead weights that minimise log loss, plus deterministic clinical overrides (e.g. "died prior to recommended surgery" → dead)
4. **Thresholding:** the decision threshold was tuned on out-of-fold predictions to maximise weighted F1.


## BAIC 2025 — Cirrhosis outcome prediction

### Problem
Predict the probability of each outcome for patients with cirrhosis:
- `C`: alive at `N_Days` (censored)
- `CL`: alive at `N_Days` because of a liver transplant
- `D`: deceased at `N_Days`

The features are 18 clinical variables, including `Drug` (D-penicillamine vs placebo), demographics, physical signs (ascites, hepatomegaly, spiders, edema), and lab values (bilirubin, albumin, copper, prothrombin, and others). The submission gives three probability columns: `Status_C`, `Status_CL`, `Status_D`.

### Approach
1. **EDA and survival analysis:** Kaplan–Meier curves by treatment arm, and a bilirubin/albumin ratio vs prothrombin analysis.
2. **Epidemiological analysis:** relative risk, excess risk, odds ratio and attributable risk for drug vs outcome, chi-square tests, a logistic model, and a Cochran–Mantel–Haenszel test adjusting for exposure variables.
3. **Modelling:** the data contains trial patients (`Drug` recorded) and non-trial patients (`Drug` missing, and most clinical signs missing with it). We trained **two separate XGBoost models**, one per group, each on the features available to it. Both were tuned by grid search on log loss with 3-fold CV.


## Reproducing
Open a notebook on Kaggle, attach the matching competition dataset, and run all cells. Main dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `statsmodels`, `lifelines`, `scipy`, `matplotlib`, `seaborn`.
