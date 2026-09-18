"""Exact complex128 full-space QAOA statevector (numpy).

Parameters are interleaved x = [gamma_1, beta_1, ..., gamma_p, beta_p].
|psi_0> = |+>^m;  layer k: psi <- exp(-i gamma_k H) psi,  psi <- exp(-i beta_k H_M) psi.
H_M is applied through the normalised Walsh-Hadamard transform W (W^2 = I):
exp(-i beta H_M) = W diag(exp(-i beta eigen)) W. This is a simulation algorithm; the physical
circuit is the product of commuting Pauli-X-string rotations.
"""
from __future__ import annotations
import numpy as np


def walsh(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]; m = n.bit_length() - 1; out = a.astype(np.complex128, copy=True)
    for i in range(m):
        v = out.reshape(-1, 2, 1 << i)
        s0 = v[:, 0, :] + v[:, 1, :]; s1 = v[:, 0, :] - v[:, 1, :]
        v[:, 0, :] = s0; v[:, 1, :] = s1
    return out / np.sqrt(n)


class QAOA:
    def __init__(self, H: np.ndarray, eigen: np.ndarray):
        self.H = np.asarray(H, float); self.eigen = np.asarray(eigen, float); self.N = len(H)

    def evolve(self, x) -> np.ndarray:
        x = np.asarray(x, float); psi = np.full(self.N, 1 / np.sqrt(self.N), np.complex128)
        for k in range(len(x) // 2):
            psi = psi * np.exp(-1j * x[2 * k] * self.H)
            psi = walsh(walsh(psi) * np.exp(-1j * x[2 * k + 1] * self.eigen))
        return psi

    def value(self, x) -> float:
        psi = self.evolve(x); return float(np.real(np.vdot(psi, self.H * psi)))

    def value_gradient(self, x):
        """Adjoint gradient with inverse replay (port of depth_joint_scaling_v2/kernel.py::reverse_gradient)."""
        x = np.asarray(x, float); psi = self.evolve(x)
        f = float(np.real(np.vdot(psi, self.H * psi))); lam = self.H * psi; g = np.zeros(len(x))
        for k in range(len(x) // 2 - 1, -1, -1):
            z = walsh(psi); a = walsh(lam)
            g[2 * k + 1] = 2 * np.real(np.vdot(a, -1j * self.eigen * z))
            inv = np.exp(1j * x[2 * k + 1] * self.eigen)
            psi = walsh(z * inv); lam = walsh(a * inv)
            g[2 * k] = 2 * np.real(np.vdot(lam, -1j * self.H * psi))
            inv = np.exp(1j * x[2 * k] * self.H)
            psi = psi * inv; lam = lam * inv
        return f, g


def direct_evolve(H, terms, x) -> np.ndarray:
    """Independent reference: explicit Pauli-X-string rotations exp(-i beta w X^s), no Walsh transform."""
    H = np.asarray(H, float); N = len(H); idx = np.arange(N); psi = np.full(N, 1 / np.sqrt(N), np.complex128)
    for k in range(len(x) // 2):
        psi = psi * np.exp(-1j * x[2 * k] * H)
        for s, w in terms:     # terms commute; exp(-i b w X^s) = cos(bw) I - i sin(bw) X^s
            th = x[2 * k + 1] * w
            psi = np.cos(th) * psi - 1j * np.sin(th) * psi[idx ^ s]
    return psi
