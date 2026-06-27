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

## Data cleaning & bias awareness

Cleaning is never neutral: each decision about what to drop, keep, or correct changes the
sample a downstream model learns from. The biggest risk in this workflow is **poor cleaning
quietly biasing that sample** — so every cleaning step (§4 of the notebook) is applied in
memory, documented, and reversible rather than baked silently into the data.

Where our cleaning choices could have introduced bias, and how we avoided it:

- **The 3,054 "uncharacterized" molecules — flagged, not dropped.** These are exactly the
  molecules whose B3LYP-relaxed geometry diverged from the input GDB-17 graph — i.e.
  disproportionately strained, unusual, or reactive structures. Silently deleting them
  (the "easy" clean) would bias any model toward well-behaved molecules and quietly shrink
  coverage of the hard cases that matter most. We add a boolean `is_uncharacterized` flag
  instead, so the choice to exclude them is explicit, auditable, and made per task.
- **The 29 identical-property rows — kept.** Naively "deduplicating identical rows" would
  delete genuine, symmetry-equivalent molecules and undercount symmetric species — turning
  a real physical feature into a fake data error and biasing the distribution.
- **The doubled-frequency anomaly (433 molecules) — corrected, not discarded.** Mishandling
  it (dropping the affected molecules, or averaging the two identical halves) would distort
  the vibrational-frequency distribution or remove a specific 0.3% slice of the data.

**Dataset-level limitations (bias present at the source, independent of cleaning).** Two
caveats are worth stating because no cleaning can remove them. First, QM9's properties are
**DFT computations, not measurements**: the harmonic vibrational frequencies run
systematically ~3–4% high versus experiment, and the rotational constants are equilibrium
values — so models trained here learn *DFT-accurate* trends, which are offset from physical
reality. Second, QM9 is a **biased sample by construction** — only molecules with ≤9 heavy
atoms drawn from C/H/O/N/F, enumerated from molecular graphs — which skews it heavily toward
elongated, chain-like (prolate) structures and away from rings and discs (see §5.2). Any
model built on this data inherits both biases regardless of how cleanly it is prepared.

## How this connects to future AI work

A clean, reproducible data pipeline is the first stage of nearly every AI project. The
ingestion, cleaning, and exploration steps here produce the trustworthy, well-understood
tabular data that downstream ML and Deep Learning models depend on.

## License / attribution

QM9 data is owned by its original authors and must be cited as above when used.
