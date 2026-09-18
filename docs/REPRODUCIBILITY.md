# Reproducibility

This repository supports three levels of reproduction. Levels 1 and 2 are run by the CI workflow on every push, with levels 2 restricted to the p = 120 endpoints.

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[plot,test]"
pytest -q
```
Tested with Python 3.11, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11 and pytest 9.1.1.

## Level 1: frozen records
Check that the records are internally consistent. This covers parent hash chain, cumulative-cost identity, budget cap, credit rule, FROZEN prefix unchanged and WARM angles all active.
```bash
python scripts/verify_records.py
```

Rebuild every figure, table and number reported in the paper into any output directory, then compare the tables and numbers with `reference_outputs/`:
```bash
python scripts/make_paper_assets.py --out-dir reproduced_assets
python scripts/check_paper_assets.py --shipped reference_outputs --regenerated reproduced_assets
```
Tables and numbers must be byte-identical. The one exception is the finite-difference gradient diagnostic (`GradErr`, about 1e-10). It depends on the platform's floating-point behaviour and is checked against an absolute bound of 5e-10.

Rebuild the README overview figures:
```bash
python scripts/make_readme_assets.py        # writes docs/assets/*.png, *.svg and readme_numbers.json
```

## Level 2: state replay
Rebuild each Hamiltonian from its instance definition, evolve the frozen winner angles, and compare P_opt, P_feas and energy with the records. The tolerance is 1e-9, and the observed differences are about 1e-15.
```bash
python scripts/replay_states.py --scope endpoints     # the 18 p=120 endpoints, about 1 min
python scripts/replay_states.py --scope checkpoints   # every protocol at p in {1,4,8,16,32,64,120}
python scripts/replay_states.py --scope all           # all 1470 fits, about 1 CPU-hour
```

## Level 3: retraining (explicit opt-in, never run automatically)
```bash
python scripts/train.py --task tv2_m16_bottleneck_00 --mixer MULTI --max-depth 3     # smoke run
python scripts/train.py --task tv2_m16_bottleneck_00 --mixer MULTI --max-depth 120 --confirm-full-retraining
```
Retraining reproduces the protocol and the initial points exactly. It is not guaranteed to reproduce the original optimizer trajectories bit for bit, because the original run used Numba kernels and L-BFGS-B paths are sensitive to rounding. The frozen records in `data/records` are the reference. Full retraining has not been repeated for this release.

## Boundaries of the data
- **Training inputs.** Training used only the instance definition, the cost Hamiltonian, the mixer and the energy with its gradient. Optimal-route labels are used only in `src/rcsp_warm/evaluation.py` and the replay scripts.
- **Winners.** Winners are the lowest-energy evaluated points, never selected by P_opt. Endpoint results are the recorded p = 120 winners, not trajectory maxima.
- **Costs.** Cumulative costs are nominal credits: 1 per energy call and 3 per energy+gradient call. Layer-work is depth × credits. Fit seconds are wall time on a shared cluster.
- **Circuit resources.** The CNOT, Toffoli and ancilla columns are upper bounds from one existing constructive compilation.
- **Saved states.** The original run's saved statevectors (about 1 MB each) are not included. States are recovered by replaying the frozen angles.
- **Further documentation.** Field definitions are in `docs/DATA_DICTIONARY.md`. Provenance and snapshot hashes are in `docs/SOURCES.md`.
