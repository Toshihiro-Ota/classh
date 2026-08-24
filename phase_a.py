"""
The MIT License (MIT) Copyright (c) 2026. Toshihiro Ota
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import brentq, minimize_scalar

from scipy.special import roots_hermite


_NGH = 600
_gh_x, _gh_w = roots_hermite(_NGH)      # stable for large n (asymptotic alg.)
GH_Z = math.sqrt(2.0) * _gh_x           # nodes for z ~ N(0, 1)
GH_W = _gh_w / math.sqrt(math.pi)       # weights (sum to 1)

# Composite Gauss-Legendre grid on [-V_MAX, 0] u [0, V_MAX]: the correction
# integrands (tanh v - sgn v, log(1 + e^{-2|v|})) are non-smooth at v = 0,
# so we split the interval there and keep spectral accuracy on each half.
_NGL_HALF = 250
_V_MAX = 40.0
_glh_x, _glh_w = np.polynomial.legendre.leggauss(_NGL_HALF)
_vp = 0.5 * _V_MAX * (_glh_x + 1.0)     # nodes on (0, V_MAX)
_wp = 0.5 * _V_MAX * _glh_w
GL_V = np.concatenate([-_vp[::-1], _vp])
GL_W = np.concatenate([_wp[::-1], _wp])

B_SWITCH = 2.0                          # GH vs localized-scheme switch

# localized pieces, precomputed on GL_V (all decay like e^{-2|v|})
_TANH_CORR = np.tanh(GL_V) - np.sign(GL_V)
_SECH2 = 1.0 / np.cosh(GL_V) ** 2
_ELL = np.log1p(np.exp(-2.0 * np.abs(GL_V)))


def _log2cosh_arr(v):
    return np.abs(v) + np.log1p(np.exp(-2.0 * np.abs(v)))


def field_moments(a, b):
    """E[tanh v], E[tanh^2 v], E[log 2 cosh v] for v ~ N(a, b^2)."""
    if b < 1e-13:
        t = math.tanh(a)
        aa = abs(a)
        return t, t * t, aa + math.log1p(math.exp(-2.0 * aa))
    if b <= B_SWITCH:
        v = a + b * GH_Z
        t = np.tanh(v)
        return (float(GH_W @ t), float(GH_W @ (t * t)), float(GH_W @ _log2cosh_arr(v)))
    # ---- large-b branch: exact principal part + localized corrections ----
    s = a / (math.sqrt(2.0) * b)
    erf_s = math.erf(s)
    phi = np.exp(-0.5 * ((GL_V - a) / b) ** 2) / (math.sqrt(2.0 * math.pi) * b)
    w = GL_W * phi
    Et = erf_s + float(w @ _TANH_CORR)
    Et2 = 1.0 - float(w @ _SECH2)
    Eabs = a * erf_s + b * math.sqrt(2.0 / math.pi) * math.exp(-min(s * s, 700.0))
    Elc = Eabs + float(w @ _ELL)
    return Et, Et2, Elc


# Noise kernel M_k(q) and entropic term Psi_k(q)
def dfact(n):
    """Double factorial n!! (n!! = 1 for n <= 0)."""
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


class Kernel:
    """M_k(q) and Psi_k(q) for even k >= 2 (exact Hermite coefficients)."""

    def __init__(self, k):
        assert k >= 2 and k % 2 == 0, "k must be a positive even integer"
        self.k = k
        if k > 2:
            js, Aj = [], []
            for j in range(1, k):                     # j odd (parity of k-1)
                if (k - 1 - j) % 2 == 0:
                    c = (math.factorial(k - 1) // (math.factorial(j) * 2 ** ((k - 1 - j) // 2) * math.factorial((k - 1 - j) // 2)))
                    js.append(j)
                    Aj.append(c * c * math.factorial(j))
            self.js = np.array(js, dtype=float)
            self.Aj = np.array(Aj, dtype=float)
            # sanity: M_k(1) = (2k-3)!!
            assert abs(self.Aj.sum() - dfact(2 * k - 3)) < 1e-9 * self.Aj.sum()

    def M(self, q, beta=None):
        """Noise moment M_k(q); beta only enters for k = 2 (AGS kernel)."""
        if self.k == 2:
            D = 1.0 - beta * (1.0 - q)
            return q / (D * D)
        return float(np.sum(self.Aj * q ** self.js))

    def M_arr(self, q):
        """Vectorized M_k(q) for k > 2 (q: ndarray)."""
        return (q[:, None] ** self.js) @ self.Aj

    def Psi(self, q, beta):
        """Psi_k(q) with Psi_k' = beta^2 M_k(q) (Psi_k(0) = 0)."""
        if self.k == 2:
            D = 1.0 - beta * (1.0 - q)
            return math.log(D) - beta * q / D
        return beta * beta * float(np.sum(self.Aj * q ** (self.js + 1.0) / (self.js + 1.0)))


def Tg_line(alpha, k):
    """RS continuous SG bifurcation temperature of the m = 0 sector."""
    if alpha <= 0.0:
        return 0.0 if k > 2 else 1.0
    if k == 2:
        return 1.0 + math.sqrt(alpha)
    return dfact(k - 1) * math.sqrt(alpha)


# Equations of state: damped fixed-point solver
def solve_fp(alpha, T, k, kern, m0, q0, with_m=True,
             damp=0.6, tol=1e-11, maxit=3500):
    """Damped fixed-point iteration on (m, q). Returns (m, q, conv, step)."""
    beta = 1.0 / T
    m = float(m0) if with_m else 0.0
    q = float(q0)
    step = 1.0
    for _ in range(maxit):
        if k == 2 and beta * (1.0 - q) >= 1.0 - 1e-9:
            q = 1.0 - (1.0 - 1e-6) / beta       # keep AGS denominator > 0
        r = kern.M(q, beta)
        b = beta * math.sqrt(max(alpha * r, 0.0))
        a = beta * m ** (k - 1) if with_m else 0.0
        Et, Et2, _ = field_moments(a, b)
        mn = min(max(Et, 0.0), 1.0) if with_m else 0.0
        qn = min(max(Et2, 0.0), 1.0)
        dm, dq = mn - m, qn - q
        m += damp * dm
        q += damp * dq
        step = max(abs(dm), abs(dq))
        if step < tol:
            return m, q, True, step
        if with_m and m < 2e-4:
            return 0.0, q, True, step   # magnetization collapsed: m = 0 branch
    return m, q, False, step


def _solve_q(alpha, T, k, kern, a, q_guess):
    """
    Inner 1D solve of q = E[tanh^2(a + b(q) z)] at fixed field strength
    a = beta m^{k-1}, with b(q) = beta sqrt(alpha M_k(q)).

    For k = 2 the AGS kernel diverges at C = beta (1 - q) -> 1, so the
    physical branch lives on q in (1 - T, 1); restricting the root search
    to this domain removes the stiffness that traps naive fixed-point
    iteration in guard-induced limit cycles at low T.
    """
    beta = 1.0 / T
    if alpha <= 0.0:
        t = math.tanh(a)
        return t * t
    qlo = (1.0 - T) + 1e-10 if (k == 2 and T < 1.0) else 1e-9
    qhi = 1.0 - 1e-14

    def F(q):
        r = kern.M(q, beta)
        b = beta * math.sqrt(max(alpha * r, 0.0))
        return field_moments(a, b)[1] - q

    Flo = F(qlo)
    if Flo <= 0.0:
        # no q > qlo root: paramagnet (k > 2 / T >= 1) or boundary value
        return 0.0 if (k > 2 or T >= 1.0) else qlo
    # warm bracket around the guess, widened geometrically if needed
    q0 = min(max(q_guess, qlo + 1e-12), qhi)
    w = 0.02
    lo, hi = max(qlo, q0 - w), min(qhi, q0 + w)
    flo, fhi = F(lo), F(hi)
    tries = 0
    while not (flo > 0.0 >= fhi) and tries < 6:
        w *= 4.0
        lo, hi = max(qlo, q0 - w), min(qhi, q0 + w)
        flo, fhi = F(lo), F(hi)
        tries += 1
    if not (flo > 0.0 >= fhi):
        lo, hi = qlo, qhi
        if F(hi) > 0.0:                 # should not happen: Et2 < 1 = q
            return hi
    return brentq(F, lo, hi, xtol=1e-13, rtol=8.9e-16)


def solve_eos_robust(alpha, T, k, kern, m0, q0, tol=1e-11, maxit=900):
    """Nested solver: damped outer iteration on m, exact inner solve for q."""
    beta = 1.0 / T
    m, q = float(m0), float(q0)
    step = 1.0
    for _ in range(maxit):
        a = beta * m ** (k - 1)
        q = _solve_q(alpha, T, k, kern, a, q)
        if alpha > 0.0:
            r = kern.M(q, beta)
            b = beta * math.sqrt(max(alpha * r, 0.0))
        else:
            b = 0.0
        Et, _, _ = field_moments(a, b)
        dm = min(max(Et, 0.0), 1.0) - m
        m += 0.7 * dm
        step = abs(dm)
        if step < tol:
            return m, q, True, step
        if m < 2e-4:
            return 0.0, q, True, step   # magnetization collapsed: m = 0 branch
    return m, q, False, step


def retrieval_solution(alpha, T, k, kern, seed=None):
    """Solve the retrieval branch; returns (exists, m, q)."""
    if seed is None:
        seed = (0.98, max(0.98, 1.0 - 0.45 * T))
    m, q, conv, step = solve_fp(alpha, T, k, kern, seed[0], seed[1], True, maxit=900)
    if conv and m > 1e-3:
        return True, m, q
    # A converged fast run with m ~ 0 is NOT proof of non-existence: for
    # k = 2 the C >= 1 guard can transiently blow up b and crash m onto the
    # spurious m = 0 fixed point. Re-check with the robust nested solver.
    m, q, conv, step = solve_eos_robust(alpha, T, k, kern, seed[0], seed[1])
    exists = (conv and m > 1e-3) or ((not conv) and m > 0.2 and step < 1e-7)
    return exists, m, q


def m0_solution(alpha, T, k, kern):
    """q of the m = 0 branch: SG (q > 0) below T_g, paramagnet otherwise."""
    if alpha <= 0.0 or T >= Tg_line(alpha, k):
        return 0.0
    q0 = max(0.6, 1.0 - 0.45 * T)
    return _solve_q(alpha, T, k, kern, 0.0, q0)


def free_energy(m, q, alpha, T, k, kern):
    """beta * f (additive constants dropped)."""
    beta = 1.0 / T
    gamma = (k - 1) / k * beta
    if alpha <= 0.0:
        # alpha-proportional terms vanish (avoid evaluating Psi_2(0) = log(1-beta))
        _, _, Elc = field_moments(beta * m ** (k - 1), 0.0)
        return gamma * m ** k - Elc
    r = kern.M(q, beta)
    b = beta * math.sqrt(max(alpha * r, 0.0))
    a = beta * m ** (k - 1)
    _, _, Elc = field_moments(a, b)
    return (gamma * m ** k
            + 0.5 * alpha * beta * beta * r * (1.0 - q)
            + 0.5 * alpha * kern.Psi(q, beta)
            - Elc)


def delta_bf(alpha, T, k, kern):
    """beta*f_retrieval - beta*f_{m=0}; None if no retrieval solution."""
    ok, m, q = retrieval_solution(alpha, T, k, kern)
    if not ok:
        return None
    bfR = free_energy(m, q, alpha, T, k, kern)
    q0 = m0_solution(alpha, T, k, kern)
    bf0 = free_energy(0.0, q0, alpha, T, k, kern)
    return bfR - bf0


def spinodal_T(alpha, k, kern, Tmin, Tscale):
    """Highest T at which a retrieval solution exists (scan + bisection)."""
    ok, m, q = retrieval_solution(alpha, Tmin, k, kern)
    if not ok:
        return 0.0
    dT = Tscale / 60.0
    T, seed = Tmin, (m, q)
    Tcap = 1.10 * Tscale
    while T < Tcap:
        T2 = min(T + dT, Tcap)
        ok2, m2, q2 = retrieval_solution(alpha, T2, k, kern, seed=seed)
        if ok2:
            T, seed = T2, (m2, q2)
        else:
            lo, hi = T, T2
            for _ in range(11):
                mid = 0.5 * (lo + hi)
                okm, mm, qm = retrieval_solution(alpha, mid, k, kern, seed=seed)
                if okm:
                    lo, seed = mid, (mm, qm)
                else:
                    hi = mid
            return 0.5 * (lo + hi)
    return T


def thermo_T(alpha, k, kern, Tmin, TR):
    """Thermodynamic first-order temperature T_M(alpha); None if absent."""
    d_lo = delta_bf(alpha, Tmin, k, kern)
    if d_lo is None or d_lo >= 0.0:
        return None                       # retrieval never global at this load
    Thi = 0.995 * TR
    d_hi = delta_bf(alpha, Thi, k, kern)
    tries = 0
    while d_hi is None and tries < 8:
        Thi *= 0.99
        d_hi = delta_bf(alpha, Thi, k, kern)
        tries += 1
    if d_hi is None:
        return None
    if d_hi < 0.0:
        return TR                         # T_M merges with the spinodal

    def f(T):
        d = delta_bf(alpha, T, k, kern)
        return d if d is not None else 1.0

    return brentq(f, Tmin, Thi, xtol=1e-7, rtol=1e-9)


def alpha_c_zero_T(k):
    """Zero-T capacity (spinodal load)."""
    if k == 2:
        def negA(t):
            m = math.erf(t)
            C = 2.0 * t * math.exp(-t * t) / (math.sqrt(math.pi) * m)
            r = 1.0 / (1.0 - C) ** 2
            return -(m * m) / (2.0 * t * t * r)
    else:
        dd = dfact(2 * k - 3)

        def negA(t):
            m = math.erf(t)
            return -(m ** (2 * (k - 1))) / (2.0 * t * t * dd)

    res = minimize_scalar(negA, bounds=(0.2, 6.0), method="bounded")
    return -res.fun


def Tc0_alpha0(k):
    """Exact alpha = 0 retrieval spinodal temperature.

    k = 2: continuous transition of m = tanh(beta m) at T = 1.
    k > 2: saddle-node of m = tanh(beta m^{k-1}); eliminating beta via
    beta = atanh(m)/m^{k-1} gives (k-1)(1-m^2) atanh(m)/m = 1, then
    T = m^{k-1}/atanh(m).
    """
    if k == 2:
        return 1.0

    def g(m):
        return (k - 1) * (1.0 - m * m) * math.atanh(m) / m - 1.0

    mstar = brentq(g, 1e-8, 1.0 - 1e-13, xtol=1e-14)
    return mstar ** (k - 1) / math.atanh(mstar)


def alpha_capacity_at_T(T, k, kern, a_hi):
    """Largest alpha with a retrieval solution at temperature T (bisection)."""
    lo, hi = 0.0, a_hi
    if retrieval_solution(hi, T, k, kern)[0]:
        return hi
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if retrieval_solution(mid, T, k, kern)[0]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# Per-k pipeline: dual-sweep boundary tracing
#
# Both phase boundaries are traced as graphs alpha(T) (right edges of the
# retrieval-existence region and of the retrieval-global region), which are
# single-valued in T even where the curves are REENTRANT in alpha (low-T
# "noses" of both the spinodal and the thermodynamic line, missed by any
# single-valued T(alpha) parametrization).  The nearly flat tops, where
# alpha(T) is steep, are resolved by a complementary alpha-sweep locating
# the upper-edge temperatures at fixed alpha.  Both point sets sample the
# same curves and are merged, sorted by T.
# ----------------------------------------------------------------------
def _alpha_star_at_T(T, k, kern, a_hi):
    """Right edge (in alpha) of the region where retrieval is the global
    minimum at temperature T; 0 if retrieval is nowhere global at this T."""
    d0 = delta_bf(0.0, T, k, kern)
    if d0 is None or d0 >= 0.0:
        return 0.0
    hi = a_hi * (1.0 - 1e-3)
    dhi = delta_bf(hi, T, k, kern)
    if dhi is not None and dhi < 0.0:
        return hi                     # merges with the existence edge
    lo = 0.0
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        d = delta_bf(mid, T, k, kern)
        if d is not None and d < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_PROBE_FRACS = np.array([0.998, 0.93, 0.85, 0.75, 0.63, 0.50,
                         0.38, 0.27, 0.17, 0.09, 0.04])


def _upper_existence_T(alpha, k, kern, Tmin, Tc0):
    """Upper edge of the retrieval-existence region at fixed alpha
    (descending probes + bisection); None if retrieval never exists."""
    probes = [max(t, Tmin) for t in Tc0 * _PROBE_FRACS] + [Tmin]
    prev = 0.9995 * Tc0               # a temperature without retrieval
    for T in probes:
        ok, _, _ = retrieval_solution(alpha, T, k, kern)
        if ok:
            lo, hi = T, prev
            for _ in range(18):
                mid = 0.5 * (lo + hi)
                if retrieval_solution(alpha, mid, k, kern)[0]:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        prev = T
    return None


def _upper_TM_T(alpha, k, kern, Tmin, T2):
    """Largest T at which retrieval is the global minimum at fixed alpha
    (descending probes below the existence edge T2); None if never global."""
    probes = sorted({max(f * T2, Tmin) for f in
                     (0.999, 0.95, 0.85, 0.72, 0.58, 0.44, 0.31, 0.20, 0.11, 0.05)} | {Tmin}, reverse=True)
    prev_pos = None
    for T in probes:
        d = delta_bf(alpha, T, k, kern)
        if d is not None and d < 0.0:
            if prev_pos is None:
                return T              # merges with the existence edge
            def f(t):
                dd = delta_bf(alpha, t, k, kern)
                return dd if dd is not None else 1.0
            return brentq(f, T, prev_pos, xtol=2e-5)
        prev_pos = T
    return None


def _as_graph(points):
    """Sort (alpha, T) points by T and drop duplicate temperatures, so the
    result is a graph alpha(T) usable with np.interp."""
    pts = sorted(points, key=lambda p: p[1])
    out = []
    for a, t in pts:
        if out and abs(t - out[-1][1]) < 1e-12:
            continue
        out.append((a, t))
    return np.array(out)


def compute_phase_diagram(k, Tmin=0.01, nT=14, nalpha=22, verbose=True):
    t0 = time.time()
    kern = Kernel(k)

    ac0 = alpha_c_zero_T(k)                      # closed-form T = 0 capacity
    Tc0 = Tc0_alpha0(k)                          # exact alpha = 0 spinodal

    # ---- T-sweep: right edges alpha_R(T), alpha_*(T) ----
    Tgrid = Tmin + (0.995 * Tc0 - Tmin) * np.linspace(0.0, 1.0, nT) ** 1.5
    aR_T = np.array([alpha_capacity_at_T(T, k, kern, 1.6 * ac0)
                     for T in Tgrid])
    aG_T = np.array([_alpha_star_at_T(T, k, kern, aR)
                     for T, aR in zip(Tgrid, aR_T)])

    R_pts = list(zip(aR_T, Tgrid))
    G_pts = [(a, T) for a, T in zip(aG_T, Tgrid) if a > 0.0]

    # ---- alpha-sweep: upper edges at fixed alpha (flat tops) ----
    a_hi = 1.02 * aR_T.max()
    alphas = np.unique(np.concatenate([
        [0.0],
        a_hi * np.array([1e-3, 3e-3, 0.01, 0.03, 0.06]),
        a_hi * np.linspace(0.10, 1.0, nalpha),
    ]))
    for a in alphas:
        if a <= 0.0:
            # exact endpoints: existence up to Tc0; for k = 2 the alpha=0
            # transition is continuous, so the global-minimum edge merges
            R_pts.append((0.0, Tc0))
            if k == 2:
                G_pts.append((0.0, Tc0))
            else:
                Tm = _upper_TM_T(0.0, k, kern, Tmin, Tc0)
                if Tm is not None:
                    G_pts.append((0.0, Tm))
            continue
        T2 = _upper_existence_T(a, k, kern, Tmin, Tc0)
        if T2 is None:
            continue
        R_pts.append((a, T2))
        Tm = _upper_TM_T(a, k, kern, Tmin, T2)
        if Tm is not None:
            G_pts.append((a, Tm))

    curveR = _as_graph(R_pts)
    curveG = _as_graph(G_pts)

    # ---- landmarks ----
    alpha_M = aG_T[0]                       # T -> 0 edge of the R region
    ac_low = aR_T[0]                        # T -> 0 capacity (spinodal)
    iR = int(np.argmax(curveR[:, 0]))
    iG = int(np.argmax(curveG[:, 0]))
    noseR = tuple(curveR[iR])               # (alpha, T) of max capacity
    noseG = tuple(curveG[iG])               # (alpha, T) of max global load
    TM0 = curveG[:, 1].max()

    res = dict(k=k, kern=kern, curveR=curveR, curveG=curveG,
               ac0=ac0, ac_low=ac_low, Tc0=Tc0, TM0=TM0,
               alpha_M=alpha_M, noseR=noseR, noseG=noseG, Tmin=Tmin)
    if verbose:
        print(f"[k={k}]  T_c(0) = {Tc0:.4f}   T_M(0) = {TM0:.4f}   "
              f"alpha_c(T->0) = {ac_low:.6g}   "
              f"alpha_c^max = {noseR[0]:.6g} at T = {noseR[1]:.3f}   "
              f"alpha_M(T->0) = {alpha_M:.6g}   "
              f"alpha_M^max = {noseG[0]:.6g} at T = {noseG[1]:.3f}   "
              f"[{time.time() - t0:.1f} s]")
    return res


def load_phase_diagram(k, outdir, verbose=True):
    """Rebuild the result dict from previously saved CSV files (--replot).

    No numerics are re-run: the boundary curves and landmarks are read back
    from modelA_lines_k*.csv / modelA_landmarks_k*.csv."""
    lmpath = os.path.join(outdir, f"modelA_landmarks_k{k}.csv")
    lnpath = os.path.join(outdir, f"modelA_lines_k{k}.csv")
    lm = {}
    with open(lmpath) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, val = line.strip().split(",")
            lm[name] = float(val)
    d = np.genfromtxt(lnpath, delimiter=",")
    Ts, aR, aG = d[:, 0], d[:, 1], d[:, 2]
    Tc0, TM0 = lm["Tc0_alpha0_exact"], lm["TM0_alpha0"]
    mR = (Ts <= Tc0 + 1e-12) | (aR > 0.0)
    mG = (Ts <= TM0 + 1e-12) | (aG > 0.0)
    res = dict(k=k, kern=None,
               curveR=np.column_stack([aR[mR], Ts[mR]]),
               curveG=np.column_stack([aG[mG], Ts[mG]]),
               Tc0=Tc0, TM0=TM0,
               ac0=lm["alpha_c_T0_closed"], ac_low=lm["alpha_c_Tmin"],
               alpha_M=lm["alpha_M_Tmin"],
               noseR=(lm["alpha_R_max"], lm["T_at_alpha_R_max"]),
               noseG=(lm["alpha_M_max"], lm["T_at_alpha_M_max"]),
               Tmin=lm["Tmin"])
    if verbose:
        print(f"[k={k}]  replotted from {lnpath}")
    return res


# Unified style (shared with the Model B/C scripts)
COL_PARA, COL_RETR, COL_FROZ = "#bdd7ee", "#f8cfa4", "#c3e2ba"  # P, R, SG
COL_SP = "#c1121f"                                               # spinodal

# font sizes (enlarged for readability)
FS_TITLE, FS_LABEL, FS_TICK = 18, 17, 13
FS_LEGEND, FS_PHASE = 14, 20


def plot_panel(ax, res):
    k = res["k"]
    curveR, curveG = res["curveR"], res["curveG"]
    Tc0, TM0 = res["Tc0"], res["TM0"]
    ac, aM = res["ac_low"], res["alpha_M"]
    noseR, noseG = res["noseR"], res["noseG"]

    amax = 1.10 * noseR[0]
    Tmax = 1.10 * max(Tc0, Tg_line(amax, k))

    # dense T grid for the fills; alpha(T) graphs via interpolation
    Td = np.linspace(0.0, Tmax, 600)
    aR_d = np.interp(Td, curveR[:, 1], curveR[:, 0], left=curveR[0, 0], right=0.0)
    aG_d = np.interp(Td, curveG[:, 1], curveG[:, 0], left=curveG[0, 0], right=0.0)
    aR_d[Td > curveR[:, 1].max()] = 0.0
    aG_d[Td > TM0] = 0.0

    # equilibrium phase fills: R (alpha < alpha_M), SG (T < T_g), P (rest)
    ag = np.linspace(0.0, amax, 400)
    AG, TDg = np.meshgrid(ag, Td)
    TgA = np.array([Tg_line(a, k) for a in ag])
    phase = np.where(AG <= aG_d[:, None], 1.0, np.where(TDg < TgA[None, :], 2.0, 0.0))
    ax.contourf(AG, TDg, phase, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=[COL_PARA, COL_RETR, COL_FROZ])

    # metastable retrieval band M (hatched), alpha_M < alpha < alpha_R
    ax.fill_betweenx(Td, aG_d, aR_d, where=aR_d > aG_d, facecolor="none", hatch="////", edgecolor=COL_SP, linewidth=0.0)

    # boundary curves (unified styles)
    ax.plot(curveG[:, 0], curveG[:, 1], "-", color="k", lw=2.0)      # T_M
    ax.plot(curveR[:, 0], curveR[:, 1], "-", color=COL_SP, lw=2.4)   # T_R
    ax.plot(ag, TgA, ls=":", color="k", lw=2.0)                      # T_g

    # landmark symbols
    # ax.plot([0], [Tc0], "o", color=COL_SP, ms=6, zorder=5)
    # ax.plot([ac], [0], "s", color=COL_SP, ms=6, zorder=5)
    # ax.plot([aM], [0], "^", color="k", ms=7, zorder=5)
    # if noseR[0] > 1.004 * ac:
    #     ax.plot([noseR[0]], [noseR[1]], "<", color=COL_SP, ms=6, zorder=5)
    # if noseG[0] > 1.004 * aM:
    #     ax.plot([noseG[0]], [noseG[1]], "<", color="k", ms=6, zorder=5)

    # region labels at region medians (same convention as Model B), with
    # per-panel manual nudges (fractions of the axis spans) keeping the
    # labels clear of the T_R / T_g curves
    NUDGE = {(4, "P"): (0.18, 0.02), (6, "P"): (0.20, 0.02),
             (4, "R"): (0.00, -0.07), (6, "R"): (0.00, 0.07),
             (4, "SG"): (0.11, 0.20), (6, "SG"): (0.23, 0.15)}
    for idx, name in ((0.0, "P"), (1.0, "R"), (2.0, "SG")):
        msk = phase == idx
        if msk.sum() < 50:
            continue
        dx, dy = NUDGE.get((k, name), (0.0, 0.0))
        ax.text(np.median(AG[msk]) + dx * amax,
                np.median(TDg[msk]) + dy * Tmax, name,
                fontsize=FS_PHASE, fontweight="bold",
                ha="center", va="center")
    mskM = (AG > aG_d[:, None]) & (AG <= aR_d[:, None])
    if mskM.sum() >= 50:
        ax.text(np.median(AG[mskM]), np.median(TDg[mskM]), "M",
                fontsize=FS_PHASE - 1, fontweight="bold", color=COL_SP,
                ha="center", va="center")

    # annotation box
    # txt = (rf"$T_c(0)={Tc0:.3f}$,  $T_M(0)={TM0:.3f}$" + "\n"
    #        + rf"$\alpha_c(0)\simeq{ac:.3g}$,  "
    #        + rf"$\alpha_c^{{\max}}\simeq{noseR[0]:.3g}$" + "\n"
    #        + rf"$\alpha_M(0)\simeq{aM:.3g}$,  "
    #        + rf"$\alpha_M^{{\max}}\simeq{noseG[0]:.3g}$")
    # ax.text(0.02, 0.985, txt, transform=ax.transAxes, fontsize=8.5,
    #         va="top", ha="left",
    #         bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    ax.set_xlim(0.0, amax)
    ax.set_ylim(0.0, Tmax)
    ax.set_xlabel(r"$\alpha_k$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$T = 1/\beta_k$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    # panel titles "(a) k=2", ... are set in main()
    # ax.set_title(rf"Model A,  $k={k}$")
    if k >= 6:
        ax.ticklabel_format(style="sci", scilimits=(-3, 3), axis="x")
        ax.xaxis.get_offset_text().set_fontsize(FS_TICK)


def legend_handles():
    """Figure-level legend (unified across the Model A/B/C scripts)."""
    return [
        Patch(fc=COL_PARA, label="P (paramagnet, $m=q=0$)"),
        Patch(fc=COL_RETR, label="R (retrieval = global minimum)"),
        Patch(fc=COL_FROZ, label="SG (RS spin glass)"),
        Patch(fc="none", hatch="////", ec=COL_SP, label="M (metastable retrieval)"),
        Line2D([], [], color="k", lw=2.0,
               label=r"$T_M(\alpha)$: first-order boundary (retrieval = global min. edge)"),
        Line2D([], [], color=COL_SP, lw=2.4,
               label=r"$T_R(\alpha)$: retrieval spinodal (existence edge)"),
        Line2D([], [], color="k", lw=2.0, ls=":",
               label=r"$T_g(\alpha)$: RS spin-glass line (continuous)"),
    ]


def add_fig_legend(fig, handles, fontsize=FS_LEGEND, max_ncol=4):
    """Figure-level legend below the panels; the number of columns is
    reduced until the legend is no wider than the plot area."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw = fig.get_window_extent().width
    leg = None
    for ncol in range(max_ncol, 0, -1):
        if leg is not None:
            leg.remove()
        leg = fig.legend(handles=handles, loc="upper center", ncol=ncol,
                         fontsize=fontsize, frameon=False,
                         bbox_to_anchor=(0.5, -0.01))
        if leg.get_window_extent(renderer).width <= 0.99 * fw:
            break
    return leg


def save_csv(res, outdir):
    k = res["k"]
    curveR, curveG = res["curveR"], res["curveG"]
    Ts = np.unique(np.concatenate([curveR[:, 1], curveG[:, 1]]))
    aR = np.interp(Ts, curveR[:, 1], curveR[:, 0], left=curveR[0, 0], right=0.0)
    aG = np.interp(Ts, curveG[:, 1], curveG[:, 0], left=curveG[0, 0], right=0.0)
    aG[Ts > res["TM0"]] = 0.0
    Tg = np.array([Tg_line(a, k) for a in aR])
    path = os.path.join(outdir, f"modelA_lines_k{k}.csv")
    hdr = ("T, alpha_R(T) [retrieval existence edge], "
           "alpha_M(T) [retrieval global-minimum edge], "
           "T_g(alpha_R) [RS SG line at alpha_R]")
    np.savetxt(path, np.column_stack([Ts, aR, aG, Tg]),
               delimiter=",", header=hdr)
    return path


def save_landmarks(res, outdir):
    """Special points of one panel (name, value rows)."""
    k = res["k"]
    noseR, noseG = res["noseR"], res["noseG"]
    lm = dict(k=k, Tc0_alpha0_exact=res["Tc0"], TM0_alpha0=res["TM0"],
              alpha_c_T0_closed=res["ac0"], alpha_c_Tmin=res["ac_low"],
              alpha_M_Tmin=res["alpha_M"],
              alpha_R_max=noseR[0], T_at_alpha_R_max=noseR[1],
              alpha_M_max=noseG[0], T_at_alpha_M_max=noseG[1],
              Tmin=res["Tmin"])
    path = os.path.join(outdir, f"modelA_landmarks_k{k}.csv")
    with open(path, "w") as f:
        f.write(f"# Model A landmarks at k = {k} (name, value)\n")
        for name, val in lm.items():
            f.write(f"{name},{val:.12g}\n")
    return path


# Self-test of the integrator
def selftest():
    from scipy.integrate import quad
    print("Integrator self-test (field_moments vs adaptive quadrature):")
    cases = [(0.5, 0.5), (1.0, 1.9), (2.0, 2.1), (0.3, 25.0),
             (5.0, 60.0), (0.0, 3.0), (30.0, 2.5), (0.7, 0.0)]
    worst = 0.0
    for a, b in cases:
        got = field_moments(a, b)
        if b < 1e-13:
            ref = (math.tanh(a), math.tanh(a) ** 2, abs(a) + math.log1p(math.exp(-2 * abs(a))))
        else:
            phi = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
            f1 = lambda z: phi(z) * math.tanh(a + b * z)
            f2 = lambda z: phi(z) * math.tanh(a + b * z) ** 2
            f3 = lambda z: phi(z) * (abs(a + b * z) + math.log1p(math.exp(-2 * abs(a + b * z))))
            ref = tuple(quad(f, -12, 12, limit=400)[0] for f in (f1, f2, f3))
        err = max(abs(g - r) for g, r in zip(got, ref))
        worst = max(worst, err)
        print(f"  a={a:6.2f} b={b:6.2f}  max|err| = {err:.2e}")
    print(f"  worst error: {worst:.2e}")
    # kernel sanity
    for k in (4, 6, 8, 10):
        kk = Kernel(k)
        assert abs(kk.M(1.0, 1.0) - dfact(2 * k - 3)) < 1e-6 * dfact(2 * k - 3)
    print("  Kernel check M_k(1) = (2k-3)!! passed for k = 4, 6, 8, 10.")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="RS finite-temperature phase diagram of Model A "
                    "(class-H dense associative memories).")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 6],
                    help="even hidden exponents k (default: 2 4 6)")
    ap.add_argument("--Tmin", type=float, default=0.01,
                    help="lowest temperature used as the T -> 0 proxy")
    ap.add_argument("--nalpha", type=int, default=22,
                    help="number of alpha grid points per k")
    ap.add_argument("--out", type=str, default=".",
                    help="output directory")
    ap.add_argument("--replot", action="store_true",
                    help="re-render the figure from the CSV files previously saved in --out (no numerics are re-run)")
    ap.add_argument("--selftest", action="store_true",
                    help="run integrator self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    for k in args.k:
        if k < 2 or k % 2:
            sys.exit(f"error: k = {k} is not a positive even integer >= 2")

    os.makedirs(args.out, exist_ok=True)

    if args.replot:
        results = [load_phase_diagram(k, args.out) for k in args.k]
    else:
        results = [compute_phase_diagram(k, Tmin=args.Tmin, nalpha=args.nalpha)
                   for k in args.k]

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.2))
    if n == 1:
        axes = [axes]
    for i, (ax, res) in enumerate(zip(axes, results)):
        plot_panel(ax, res)
        ax.set_title(rf"({chr(97 + i)})  $k={res['k']}$", fontsize=FS_TITLE)
    fig.tight_layout()
    add_fig_legend(fig, legend_handles())
    # fig.suptitle("Model A (Ising visible spins): "
    #              "RS finite-temperature phase diagram", fontsize=12.5)
    tag = "_".join(f"k{k}" for k in args.k)
    png = os.path.join(args.out, f"modelA_phase_diagram_{tag}.png")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    print(f"figure  -> {png}")
    if not args.replot:                    # in replot mode the CSVs are inputs
        for res in results:
            print(f"csv     -> {save_csv(res, args.out)}")
            print(f"csv     -> {save_landmarks(res, args.out)}")


if __name__ == "__main__":
    main()
