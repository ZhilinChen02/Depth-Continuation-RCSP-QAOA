# Sources and provenance

- **Study records.** All records come from one completed run of the author's research repository: the fixed-m16 continuation run `v2_004_m16_complete`. It contains 1470 of 1470 fits, with WARM and FROZEN at every depth from 1 to 120 and RANDOM at the checkpoints. The export script read the original files without modifying them.
  - The original summary (`ALL_POINTS.csv`) has SHA-256 `f23369543820d0bd6080e997bca69a0cc756df5c6943d39c11451f457220c717`.
- **Snapshot identity.** The run is revision r04. It has 6829 files, and the aggregate SHA-256 of the sorted per-file hashes is
  `8593fe7655e3a14a39e03585ab597e50e95c633f4e35808f4f6df7e3c9b4b795`.
  - A separately kept archive copy of the run, with SHA-256 `f21099581d0d2d9b728ac5aa4345498c1e67f7069d80e54ae1c5cbc43848a867`, is byte-identical member by member.
  - The export in `data/records` was produced from this snapshot.
- **Code.** The code is a clean single-package port of the original implementation modules: instance generator rules, forest cost transform, route predicate, graph masks, numba simulator and adjoint, and trainer. The ported functions reproduce the original training arrays bit for bit, every recorded initial vector, and the saved states to within 1.3e-15.
- **Not included.**
  - The original saved statevectors (about 1 MB each).
  - Per-call journals.
  - Instances, depths and seeds of the wider campaign that were not completed to p = 120.
  - Unrelated research branches: other representations and objectives, other mixers or oracles, and Max-Cut experiments.
- **Third-party code.** None is vendored. Dependencies are NumPy, SciPy, Matplotlib and pytest, installed from PyPI.
- **License.** Not yet confirmed by the author, so no LICENSE file is included (see README).
