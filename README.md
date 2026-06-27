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

## Future integration reflections

This project is deliberately the **bottom rung of a ladder**. The long-term goal is to
predict the **vibrational/IR spectra of aromatic molecules at chemical accuracy
(~1 kcal/mol, CCSD(T)-quality) without running a per-molecule supercomputer calculation**.
QM9 is the right place to *start* — it is cheap, large, tabular, and already exposes the
spectra-relevant quantities (the per-mode frequencies table and the dipole moment `mu`) —
even though its DFT accuracy means it is a *pre-training* tier, not the final target.

### How the workflow changes when we move into Machine Learning

The shift is from *describing* the data to *predicting* from it. Concretely:

- **Define a learning target.** Pick property column(s) to predict (e.g. `gap`, `mu`, or a
  vibrational frequency) from structural features, rather than just summarizing them.
- **Split before touching the data.** Add a train / validation / test split (using the
  fixed seed `42` already established here) and fit *all* preprocessing on the training
  fold only, to avoid leakage.
- **Scale and featurize.** QM9's columns span wildly different magnitudes (rotational
  constant `A` reaches hundreds of thousands of GHz while `gap` is ~0.25 Ha), so features
  need normalizing; molecular structure must be turned into model-ready features (atom/bond
  descriptors from the per-atom table, or SMILES-derived fingerprints).
- **Decide what the flags mean for training.** The `is_uncharacterized` flag and the
  documented cleaning choices become explicit modeling decisions (include, exclude, or
  weight those 3,054 molecules) instead of silent data edits.
- **Measure, don't just look.** Replace descriptive EDA with quantitative metrics (MAE/RMSE)
  and cross-validation.

Crucially, ML also makes QM9's **DFT ceiling** visible: a model can only be as accurate as
its labels, so DFT-trained predictions inherit the ~3–4% frequency offset. That limitation
is what motivates the deep-learning and transfer-learning steps below.

### How the data should be prepared for neural networks

Tabular regression treats each molecule as a flat row, but spectra come from *structure*, so
deep learning wants a **structural representation**:

- Represent each molecule as a **graph** — nodes = atoms (carrying element, charge, and
  geometry features from the per-atom table), edges = bonds — so a graph neural network can
  learn directly from connectivity instead of hand-built columns.
- Handle **variable molecule size** (QM9 ranges from 3 to 29 atoms) with padding/masking and
  batched tensors suitable for a GPU data loader; normalize the regression targets.
- Frame the network to learn the **potential-energy surface (PES)**, not to memorize spectra.
  This is the key physical idea that ties the whole project together:

  > **Why spectra live inside the energy surface.** The PES is the molecule's energy
  > `E(x)` as a function of where its nuclei sit. A stable molecule rests at the bottom of a
  > valley (a minimum), where the forces — the gradient `∇E` — are zero. A vibration is the
  > molecule rocking inside that valley, and how fast it rocks depends on how **curved** the
  > valley walls are: steep, stiff directions vibrate fast (high frequency), shallow
  > directions vibrate slowly. That curvature in every direction at once is the **Hessian**,
  > the matrix of second derivatives `H_ij = ∂²E / ∂x_i ∂x_j`. To turn curvature into
  > frequencies you **mass-weight** the Hessian (heavy atoms swing slower) and
  > **diagonalize** it: each **eigenvalue** gives a frequency `ω = √(λ/μ)` — the familiar
  > spring relation, frequency ∝ √(stiffness / mass) — and each **eigenvector** is a
  > **normal mode**, the specific pattern of atoms moving together (a stretch, a bend, a ring
  > breathing mode). IR **intensities** then come from how the molecule's **dipole** changes
  > along each mode. So a network that reproduces the energy surface (plus a small dipole
  > model) yields the full spectrum for free — no per-molecule supercomputer run.

- Adopt **transfer learning** to escape the DFT ceiling: pre-train on the large, cheap
  DFT-level data (QM9-/ANI-1x-scale), then fine-tune on a small, expensive
  **CCSD(T)-accurate** set (e.g. the ANI-1ccx philosophy, ~500k conformations at
  CCSD(T)*/CBS). QM9 here is the bottom rung — the cheap pre-training tier the high-accuracy
  fine-tuning depends on.

### The potential for agentic AI automation

Much of this pipeline is rule-driven and verifiable, which makes it well suited to an
**agentic** workflow:

- **Self-validating ingestion.** An agent can auto-detect and re-ingest new dataset releases,
  re-run the validation suite, and flag anomalies — exactly the kind of doubled-frequency
  defect that was caught and corrected by hand here.
- **Documented, auditable cleaning.** An agent can propose cleaning actions *with
  justifications* (flag vs. drop vs. correct) and regenerate the figures and summary tables,
  keeping the "cleaning is never neutral" discipline of §4 intact.
- **Active-learning loop.** The most powerful step: an agent drives the transfer-learning
  ladder itself — pre-train on cheap DFT data, identify the molecules where the model is
  *least certain*, queue new high-accuracy (CCSD(T)) calculations only for those, fold the
  results back in, and re-derive spectra. That turns "predict spectra without a
  supercomputer" into a closed loop that spends expensive compute only where it most improves
  the model.

We will revisit and refine these reflections as the subsequent Machine Learning, Deep
Learning, Generative AI, and Agentic AI projects build on this foundation.

## License / attribution

QM9 data is owned by its original authors and must be cited as above when used.
