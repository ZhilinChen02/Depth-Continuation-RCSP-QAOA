"""Access to the frozen study records shipped under data/."""
from __future__ import annotations
import csv, gzip, json
from pathlib import Path
import numpy as np
from .problem import cost_hamiltonian
from .mixers import spectrum, mixer_terms
from .simulator import QAOA

DATA = Path(__file__).resolve().parents[2] / "data"
UNITS = [f"{t}__{m}__s9101" for t in ["tv2_m16_branch_00", "tv2_m16_cycle_00", "tv2_m16_bottleneck_00"] for m in ["X", "MULTI"]]


def data_dir(root=None) -> Path:
    return Path(root) if root else DATA


def task(task_id, root=None) -> dict:
    return json.loads((data_dir(root) / "instances" / f"{task_id}.json").read_text())


def fits(root=None) -> list[dict]:
    with open(data_dir(root) / "records" / "fits.csv") as f:
        return list(csv.DictReader(f))


def angles(root=None) -> dict:
    with gzip.open(data_dir(root) / "records" / "angles.json.gz", "rt") as f:
        return json.load(f)


def build(task_id, mixer, root=None):
    t = task(task_id, root); ch = cost_hamiltonian(t)
    return t, ch, QAOA(ch.H, spectrum(t, mixer)), mixer_terms(t, mixer)
