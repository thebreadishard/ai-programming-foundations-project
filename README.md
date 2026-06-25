# AI Programming Foundations — QM9 Reproducible Data Workflow

A clean, reproducible data workflow built on the **QM9** quantum-chemistry dataset, using
only core Python data tools (NumPy, pandas, Matplotlib/Seaborn, Jupyter). This project is
the foundation that later Machine Learning, Deep Learning, Generative AI, and Agentic AI
work will build on — the focus here is the **workflow**, not model training.

## Dataset

**QM9** — 133,885 small organic molecules (up to 9 heavy atoms: C, O, N, F) with 15
quantum-chemical properties computed at the B3LYP/6-31G(2df,p) level of theory.

- **Source:** Figshare collection 978904
- **Citation:** Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014).
  *Quantum chemistry structures and properties of 134 kilo molecules.* Scientific Data, 1, 140022.

The 15 properties (line 2 of each `.xyz` file): `A, B, C` (rotational constants), `mu`
(dipole moment), `alpha` (polarizability), `HOMO, LUMO, gap`, `R2` (spatial extent),
`zpve`, `U0, U, H, G` (energies/enthalpy/free energy), `Cv` (heat capacity).

## Project structure

```
.
├─ notebooks/
│  └─ qm9_workflow.ipynb     # main analysis: ingest -> clean -> EDA -> visualize -> findings
├─ data/
│  ├─ raw/                   # original QM9 download (git-ignored, read-only)
│  └─ processed/             # tidy, analysis-ready outputs (git-ignored)
├─ reports/figures/          # exported plots
├─ requirements.txt          # top-level dependencies
├─ requirements-lock.txt     # fully pinned environment
└─ .gitignore
```

## Setup

Built and tested on **Python 3.14**.

```powershell
# 1. Create and activate a virtual environment
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Register the Jupyter kernel and launch
python -m ipykernel install --user --name qm9-capstone --display-name "Python 3.14 (QM9 .venv)"
jupyter lab
```

Then open `notebooks/qm9_workflow.ipynb` and select the **Python 3.14 (QM9 .venv)** kernel.

## Reproducibility

- Exact runtime and library versions are printed in the notebook's setup cell.
- A fixed random seed (`42`) is set for any sampling/splitting.
- `requirements-lock.txt` captures the full pinned dependency tree.

## How this connects to future AI work

A clean, reproducible data pipeline is the first stage of nearly every AI project. The
ingestion, cleaning, and exploration steps here produce the trustworthy, well-understood
tabular data that downstream ML and Deep Learning models depend on.

## License / attribution

QM9 data is owned by its original authors and must be cited as above when used.
