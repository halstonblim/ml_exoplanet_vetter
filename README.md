# Machine Learning Exoplanet Vetter

A research‑grade toolkit for **automatically vetting transiting‑exoplanet candidates**—with both
feature‑based models (Random Forest, Logistic Regression) and deep‑learning models
(Astronet‑style CNNs).  
The code was originally written to sift through **TESS sectors 1‑22** light curves and
recover the _Kepler_‑inspired planet candidate labels released by NASA.

| Section | What you’ll find |
|---------|------------------|
| [`getLCFeatures.py`](getLCFeatures.py) | End‑to‑end feature extraction from pre‑processed `.h5` light‑curves. |
| [`baseline.py`](baseline.py) | Classic ML baselines (Random Forest + Linear/Logistic Regression) with k‑fold CV utilities. |
| `CNN/` & `Modules/astronet/` | A lightly‑modified port of the **Astronet** convolutional architecture. |
| `Notebooks/` | Exploratory notebooks and prototype experiments. |
| `Data/` | **Sample** light‑curve sectors and planet label TSV for quick testing (full TESS download not included). |

---

## 1 .  Quick‑start

```bash
# ➤ 1. Clone and create a fresh environment
git clone https://github.com/halstonblim/ml_exoplanet_vetter.git
cd ml_exoplanet_vetter
python -m venv .venv && source .venv/bin/activate

# ➤ 2. Install core requirements
pip install numpy scipy scikit-learn h5py tqdm matplotlib
# (Optional) CNN:  pip install "tensorflow<2.0"  # Astronet is TF‑1.x compatible

# ➤ 3. Extract features for Sector 1 (as an example)
python getLCFeatures.py  \
    --data_dir Data/2020_03_23_TestData \
    --sector sector-1

# ➤ 4. Train the Random‑Forest baseline
python - <<'PY'
from baseline import *
x, y = loadData(['sector-1'], datapath='Data')
results, _ = testModel(x, y, RFModel, depth=12, k=5)
print('Mean precision‑recall:', aggregateScores(results).shape)
PY
```

> **Note**  
> The repository ships only a **tiny test subset** of TESS light curves so the above runs instantly.  
> For a full science run you will need to download and place the pre‑processed
> `.h5` files for each sector in the same directory layout.

---

## 2 .  Feature set

`getLCFeatures.py` computes 25 astrophysical and light‑curve statistics per
candidate, including:

| Index | Description |
|-------|-------------|
| `0` | Astronet CNN score |
| `1‑3 / 7‑9` | Transit depth (global / local views, best aperture ± 1) |
| `4‑6 / 10‑12` | Corresponding uncertainty estimates |
| `13‑18` | Even/odd transit symmetry + secondary‑eclipse depth |
| `19‑24` | Stellar identifiers & TESS mag / log g / mass / radius / T_eff |

See the in‑line comments for the complete mapping

---

## 3 .  Classical ML baselines

The baseline script exposes helper functions to:

* **Load** all sector feature matrices (`loadData`)  
* **Balance** the class distribution via positive‑class over‑sampling (`overSamplePos`)  
* **Train** a model with stratified K‑fold cross‑validation (`testModel`)  
* **Produce** precision‑recall curves (`getPRCurve`)  

Current models:

* `RFModel` – Random Forest with tunable depth  
* `LRModel` – Linear/Logistic Regression baseline  

Implementation details live in [`baseline.py`](baseline.py). 

---

## 4 .  Deep‑learning model (Astronet)

The original **Astronet** CNN (Shallue & Vanderburg 2018) is included under
`Modules/astronet/` (median‑filter helper shown here).  
It remains TF‑1.x for historical reproducibility.  
The CNN training loops are located in `CNN/exoplanet/` and were used to
benchmark against the feature baselines.

---

## Repository layout

```
.
├── baseline.py
├── getLCFeatures.py
├── Data/                      # ⇢ sample sector & label TSV
├── CNN/                       # ⇢ Astronet training & checkpoints
├── LogisticRegression/        # ⇢ quick‑and‑dirty notebook model
├── Modules/
│   └── astronet/              # ⇢ TF ops & utilities
└── Notebooks/                 # ⇢ exploratory analysis
```
