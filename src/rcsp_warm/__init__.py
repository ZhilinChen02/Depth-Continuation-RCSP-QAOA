"""rcsp_warm: exact full-space QAOA for small resource-constrained shortest-path (RCSP) instances,
with the WARM / FROZEN / RANDOM training protocols studied in the accompanying paper.

Module boundaries
-----------------
problem      public instance data, route predicate, cost Hamiltonian (training-visible)
mixers       X and graph multi-X mixer spectra (training-visible, topology only)
simulator    exact complex128 statevector, energy and adjoint gradient
protocols    initialisation rules and the budgeted L-BFGS-B fit (training side)
evaluation   optimal-route labels and probability metrics (evaluation side only;
             never imported by the training modules)
records      loading of the frozen records shipped in data/
"""
__version__ = "1.0.0"
