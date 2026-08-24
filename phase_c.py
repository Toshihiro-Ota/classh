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


# Noise kernel (identical to Model A: the crosstalk moment is universal)
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


# Closed-form landmarks
def Tc0_alpha0(k):
    """Exact alpha = 0 retrieval spinodal temperature."""
    if k == 2:
        return 1.0                                # continuous transition
    return (2.0 / k) * ((k - 2.0) / k) ** ((k - 2) / 2.0)


def alpha_c_zero_T(k):
    """Exact T = 0 capacity; zero for k = 2 (marginality)."""
    if k == 2:
        return 0.0
    return (k - 2) ** (k - 2) / ((k - 1) ** (k - 1) * dfact(2 * k - 3))


def alpha_M_zero_T(k):
    """Exact T = 0 endpoint of the thermodynamic line."""
    if k == 2:
        return 0.0
    return (k ** (k - 2) * (k - 2) ** (k - 2)
            / ((k - 1) ** (2 * k - 2) * dfact(2 * k - 3)))


def TM0_alpha0(k):
    """Exact alpha = 0 thermodynamic transition temperature (k > 2).

    On the alpha = 0 retrieval branch q = m^2, T = (1-m^2) m^{k-2}, the
    free-energy difference to the paramagnet (beta f_P = 0) is
        Delta = -m^2 / (k (1-m^2)) - (1/2) log(1-m^2);
    the root on the stable branch m^2 in ((k-2)/k, 1) gives T_M(0).
    For k = 2 the alpha = 0 transition is continuous: T_M(0) = 1.
    """
    if k == 2:
        return 1.0

    def delta_of_x(x):                            # x = m^2
        return -x / (k * (1.0 - x)) - 0.5 * math.log(1.0 - x)

    xlo = (k - 2.0) / k + 1e-12                   # spinodal endpoint
    xhi = 1.0 - 1e-12
    xstar = brentq(delta_of_x, xlo, xhi, xtol=1e-14)
    return (1.0 - xstar) * xstar ** ((k - 2) / 2.0)


# Branch solvers (all algebraic)
def retrieval_solution(alpha, T, k, kern, nscan=400):
    """Solve the retrieval branch; returns (exists, m, q).

    For k > 2 the signal equation gives 1 - q = T m^{2-k} exactly, and
    retrieval is the largest root of G(m). The stable branch is the
    largest root (saddle-node structure: G < 0 at both domain ends,
    G > 0 between the root pair when it exists).
    For k = 2 retrieval exists only at alpha = 0 (m^2 = 1 - T, T < 1).
    """
    if k == 2:
        if alpha <= 0.0 and T < 1.0:
            m2 = 1.0 - T
            return True, math.sqrt(m2), m2
        return False, 0.0, 0.0
    if T >= 1.0:
        return False, 0.0, 0.0
    mlo = T ** (1.0 / (k - 2))
    if mlo >= 1.0:
        return False, 0.0, 0.0

    def G(m):
        q = 1.0 - T * m ** (2.0 - k)
        return (q - m * m) * m ** (2 * k - 4) - alpha * kern.M(q)

    ms = np.linspace(mlo * (1.0 + 1e-12), 1.0, nscan)
    qs = 1.0 - T * ms ** (2.0 - k)
    Gs = (qs - ms * ms) * ms ** (2 * k - 4) - alpha * kern.M_arr(qs)
    i = int(np.argmax(Gs))
    if Gs[i] <= 0.0:
        return False, 0.0, 0.0
    # largest root: G > 0 at ms[i], G(1) = -T - alpha M(1-T) < 0
    mstar = brentq(G, ms[i], 1.0, xtol=1e-14)
    return True, mstar, 1.0 - T * mstar ** (2.0 - k)


def alpha_R_at_T(T, k, kern, nscan=600):
    """Retrieval capacity at temperature T:
    alpha_R(T) = max_m (q - m^2) m^{2k-4} / M_k(q), q = 1 - T m^{2-k}.
    Returns 0 if no retrieval exists at this T (k = 2, or T >= Tc0)."""
    if k == 2 or T >= 1.0:
        return 0.0
    mlo = T ** (1.0 / (k - 2))
    if mlo >= 1.0:
        return 0.0
    ms = np.linspace(mlo * (1.0 + 1e-12), 1.0 - 1e-12, nscan)
    qs = 1.0 - T * ms ** (2.0 - k)
    Ms = kern.M_arr(qs)
    with np.errstate(divide="ignore", invalid="ignore"):
        As = np.where(Ms > 1e-300, (qs - ms * ms) * ms ** (2 * k - 4) / Ms, -np.inf)
    i = int(np.argmax(As))
    if not np.isfinite(As[i]) or As[i] <= 0.0:
        return 0.0

    def negA(m):
        q = 1.0 - T * m ** (2.0 - k)
        Mq = kern.M(q)
        if Mq <= 1e-300:
            return np.inf
        return -(q - m * m) * m ** (2 * k - 4) / Mq

    lo = ms[max(i - 1, 0)]
    hi = ms[min(i + 1, nscan - 1)]
    res = minimize_scalar(negA, bounds=(lo, hi), method="bounded", options={"xatol": 1e-13})
    return max(-res.fun, As[i], 0.0)


def m0_solution(alpha, T, k, kern):
    """q of the m = 0 branch: RS SG (q > 0) below T_g, paramagnet above.

    k = 2: the noise equation gives beta (1-q) = 1/(1+sqrt(alpha)) exactly,
    i.e. q = 1 - T/(1+sqrt(alpha)) below T_g = 1 + sqrt(alpha).
    k > 2: largest root of h(q) = alpha beta^2 M_k(q) (1-q)^2 - q, the
    branch continuously connected to the T -> 0 frozen solution (same
    convention as the Model A implementation).
    """
    if alpha <= 0.0 or T >= Tg_line(alpha, k):
        return 0.0
    if k == 2:
        return 1.0 - T / (1.0 + math.sqrt(alpha))
    beta = 1.0 / T
    c = alpha * beta * beta

    def h(q):
        return c * kern.M(q) * (1.0 - q) ** 2 - q

    qs = np.concatenate([np.geomspace(1e-9, 0.05, 80), np.linspace(0.05, 1.0 - 1e-12, 500)])[::-1]
    hs = c * kern.M_arr(qs) * (1.0 - qs) ** 2 - qs
    pos = np.where(hs > 0.0)[0]
    if len(pos) == 0:
        return 0.0
    i = pos[0]                                    # largest q with h > 0
    hi = qs[i - 1] if i > 0 else 1.0 - 1e-13      # h(hi) < 0
    return brentq(h, qs[i], hi, xtol=1e-15)


def free_energy(m, q, alpha, T, k, kern):
    """beta * f (additive constants dropped)."""
    beta = 1.0 / T
    val = (-(beta / k) * m ** k
           - 0.5 * (math.log(1.0 - q) + (q - m * m) / (1.0 - q)))
    if alpha > 0.0:
        val += 0.5 * alpha * kern.Psi(q, beta)
    return val


def delta_bf(alpha, T, k, kern):
    """beta (f_retrieval - f_{m=0}); None if no retrieval solution."""
    ok, m, q = retrieval_solution(alpha, T, k, kern)
    if not ok:
        return None
    q0 = m0_solution(alpha, T, k, kern)
    return (free_energy(m, q, alpha, T, k, kern) - free_energy(0.0, q0, alpha, T, k, kern))


def alpha_star_at_T(T, k, kern, a_hi):
    """Right edge (in alpha) of the region where retrieval is the global
    minimum at temperature T; 0 if retrieval is nowhere global at this T."""
    d0 = delta_bf(0.0, T, k, kern)
    if d0 is None or d0 >= 0.0:
        return 0.0
    hi = a_hi * (1.0 - 1e-6)
    dhi = delta_bf(hi, T, k, kern)
    if dhi is not None and dhi < 0.0:
        return hi                                 # merges with existence edge
    lo = 0.0
    for _ in range(42):
        mid = 0.5 * (lo + hi)
        d = delta_bf(mid, T, k, kern)
        if d is not None and d < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# Per-k pipeline: T-parametrized boundary tracing
# ----------------------------------------------------------------------
def compute_phase_diagram(k, Tmin=0.01, nT=160, verbose=True):
    t0 = time.time()
    kern = Kernel(k)

    Tc0 = Tc0_alpha0(k)                # exact alpha = 0 spinodal [Eq. (43)]
    ac0 = alpha_c_zero_T(k)            # exact T = 0 capacity    [Eq. (42)]
    aM0 = alpha_M_zero_T(k)            # exact T = 0 thermo endpoint
    TM0x = TM0_alpha0(k)               # exact alpha = 0 thermo temperature

    if k == 2:
        # zero capacity: retrieval exists only on the alpha = 0 axis
        res = dict(k=k, kern=kern, curveR=None, curveG=None,
                   Tc0=Tc0, TM0=TM0x, ac0=0.0, ac_low=0.0,
                   alpha_M=0.0, aM0=0.0, noseR=(0.0, 0.0),
                   noseG=(0.0, 0.0), Tmin=Tmin)
        if verbose:
            print(f"[k=2]  T_c(0) = 1 (exact, continuous at alpha = 0)   "
                  f"alpha_c = 0 (spherical marginality: no retrieval at "
                  f"any alpha > 0)   [{time.time() - t0:.1f} s]")
        return res

    # ---- T-sweep: right edges alpha_R(T), alpha_*(T) ----
    Tgrid = Tmin + (Tc0 * (1.0 - 1e-9) - Tmin) * np.linspace(0, 1, nT) ** 1.2
    aR_T = np.array([alpha_R_at_T(T, k, kern) for T in Tgrid])
    aG_T = np.array([alpha_star_at_T(T, k, kern, aR) if aR > 0 else 0.0
                     for T, aR in zip(Tgrid, aR_T)])

    # graphs alpha(T); exact T = 0 and alpha = 0 endpoints appended
    curveR = np.array([(ac0, 0.0)]
                      + [(a, T) for a, T in zip(aR_T, Tgrid)]
                      + [(0.0, Tc0)])
    curveG = np.array([(aM0, 0.0)]
                      + [(a, T) for a, T in zip(aG_T, Tgrid) if a > 0.0]
                      + [(0.0, TM0x)])

    # ---- landmarks ----
    alpha_M = aG_T[0]                  # T -> 0 (Tmin) edge of the R region
    ac_low = aR_T[0]                   # T -> 0 (Tmin) capacity
    iR = int(np.argmax(curveR[:, 0]))
    iG = int(np.argmax(curveG[:, 0]))
    noseR = tuple(curveR[iR])
    noseG = tuple(curveG[iG])

    res = dict(k=k, kern=kern, curveR=curveR, curveG=curveG,
               Tc0=Tc0, TM0=TM0x, ac0=ac0, ac_low=ac_low,
               alpha_M=alpha_M, aM0=aM0, noseR=noseR, noseG=noseG,
               Tmin=Tmin)
    if verbose:
        print(f"[k={k}]  T_c(0) = {Tc0:.6g} (exact)   "
              f"T_M(0) = {TM0x:.6g} (exact)   "
              f"alpha_c(0) = {ac0:.6g} (exact)  vs  "
              f"alpha_R(T={Tmin}) = {ac_low:.6g}   "
              f"alpha_M(0) = {aM0:.6g} (exact)  vs  "
              f"alpha_M(T={Tmin}) = {alpha_M:.6g}")
        print(f"       noses: alpha_R^max = {noseR[0]:.6g} at T = "
              f"{noseR[1]:.3f}   alpha_M^max = {noseG[0]:.6g} at T = "
              f"{noseG[1]:.3f}   [{time.time() - t0:.1f} s]")
    return res


def load_phase_diagram(k, outdir, verbose=True):
    """Rebuild the result dict from previously saved CSV files (--replot).

    No numerics are re-run: the boundary curves and landmarks are read back
    from modelC_lines_k*.csv / modelC_landmarks_k*.csv."""
    lmpath = os.path.join(outdir, f"modelC_landmarks_k{k}.csv")
    lnpath = os.path.join(outdir, f"modelC_lines_k{k}.csv")
    lm = {}
    with open(lmpath) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, val = line.strip().split(",")
            lm[name] = float(val)
    Tc0, TM0 = lm["Tc0_alpha0_exact"], lm["TM0_alpha0_exact"]
    base = dict(k=k, kern=None, Tc0=Tc0, TM0=TM0,
                ac0=lm["alpha_c_T0_exact"], aM0=lm["alpha_M_T0_exact"],
                ac_low=lm["alpha_c_Tmin"], alpha_M=lm["alpha_M_Tmin"],
                noseR=(lm["alpha_R_max"], lm["T_at_alpha_R_max"]),
                noseG=(lm["alpha_M_max"], lm["T_at_alpha_M_max"]),
                Tmin=lm["Tmin"])
    if k == 2:                               # zero-capacity panel
        res = dict(base, curveR=None, curveG=None)
    else:
        d = np.genfromtxt(lnpath, delimiter=",")
        Ts, aR, aG = d[:, 0], d[:, 1], d[:, 2]
        mR = (Ts <= Tc0 + 1e-12) | (aR > 0.0)
        mG = (Ts <= TM0 + 1e-12) | (aG > 0.0)
        res = dict(base, curveR=np.column_stack([aR[mR], Ts[mR]]), curveG=np.column_stack([aG[mG], Ts[mG]]))
    if verbose:
        print(f"[k={k}]  replotted from {lnpath}")
    return res


# Unified style (shared with the Model A/B scripts)
COL_PARA, COL_RETR, COL_FROZ = "#bdd7ee", "#f8cfa4", "#c3e2ba"  # P, R, SG
COL_SP = "#c1121f"                                               # spinodal

# font sizes (enlarged for readability)
FS_TITLE, FS_LABEL, FS_TICK = 18, 17, 13
FS_LEGEND, FS_PHASE = 14, 20


def plot_panel(ax, res):
    k = res["k"]
    Tc0, TM0 = res["Tc0"], res["TM0"]

    if k == 2:
        # zero-capacity panel: retrieval lives only on the alpha = 0 axis
        amax = 0.15
        Tmax = 1.10 * Tg_line(amax, 2)
        ag = np.linspace(0.0, amax, 400)
        Td = np.linspace(0.0, Tmax, 600)
        AG, TDg = np.meshgrid(ag, Td)
        TgA = np.array([Tg_line(a, 2) for a in ag])
        phase = np.where(TDg < TgA[None, :], 2.0, 0.0)   # SG below T_g, P above
        ax.contourf(AG, TDg, phase, levels=[-0.5, 0.5, 1.5, 2.5],
                    colors=[COL_PARA, COL_RETR, COL_FROZ])
        ax.plot(ag, TgA, ls=":", color="k", lw=2.0)      # T_g
        ax.plot([0.0, 0.0], [0.0, 1.0], "-", color=COL_SP, lw=5.0, solid_capstyle="butt")  # retrieval segment on alpha = 0
        # landmark symbol
        # ax.plot([0], [1.0], "o", color=COL_SP, ms=6, zorder=5)
        for idx, name in ((0.0, "P"), (2.0, "SG")):
            msk = phase == idx
            if msk.sum() < 50:
                continue
            ax.text(np.median(AG[msk]), np.median(TDg[msk]), name,
                    fontsize=FS_PHASE, fontweight="bold",
                    ha="center", va="center")
        # annotation box
        # txt = (r"$\alpha_c = 0$ (marginal):" + "\n"
        #        + r"$\beta_k(1-q)m^{k-2}=1$ hits the" + "\n"
        #        + r"AGS pole of $M_2$ for any $\alpha_k>0$;" + "\n"
        #        + r"retrieval only at $\alpha_k=0$, $T<1$" + "\n"
        #        + r"(crimson segment)")
        # ax.text(0.02, 0.985, txt, transform=ax.transAxes, fontsize=8.5,
        #         va="top", ha="left",
        #         bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.set_xlim(0.0, amax)
        ax.set_ylim(0.0, Tmax)
        ax.set_xlabel(r"$\alpha_k$", fontsize=FS_LABEL)
        ax.set_ylabel(r"$T = 1/\beta_k$", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        # panel titles "(a) k=2", ... are set in main()
        # ax.set_title(rf"Model C,  $k={k}$")
        return

    curveR, curveG = res["curveR"], res["curveG"]
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
    # ax.plot([res["ac0"]], [0], "s", color=COL_SP, ms=6, zorder=5)
    # ax.plot([res["aM0"]], [0], "^", color="k", ms=7, zorder=5)
    # if noseR[0] > 1.004 * res["ac0"]:
    #     ax.plot([noseR[0]], [noseR[1]], "<", color=COL_SP, ms=6, zorder=5)
    # if noseG[0] > 1.004 * res["aM0"]:
    #     ax.plot([noseG[0]], [noseG[1]], "<", color="k", ms=6, zorder=5)

    # region labels at region medians (same convention as Model B), with
    # per-panel manual nudges (fractions of the axis spans) keeping the
    # labels clear of the T_R / T_g curves
    NUDGE = {(6, "SG"): (0.07, 0.05), (6, "R"): (0.00, -0.05)}
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
    # txt = (rf"$T_c(0)={Tc0:.4g}$,  $T_M(0)={TM0:.4g}$" + "\n"
    #        + rf"$\alpha_c(0)={res['ac0']:.4g}$" + "\n"
    #        + rf"$\alpha_M(0)={res['aM0']:.4g}$")
    # if noseR[0] > 1.004 * res["ac0"]:
    #     txt += "\n" + rf"$\alpha_c^{{\max}}\simeq{noseR[0]:.3g}$"
    # if noseG[0] > 1.004 * res["aM0"]:
    #     txt += "\n" + rf"$\alpha_M^{{\max}}\simeq{noseG[0]:.3g}$"
    # ax.text(0.02, 0.985, txt, transform=ax.transAxes, fontsize=8.5,
    #         va="top", ha="left",
    #         bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    ax.set_xlim(0.0, amax)
    ax.set_ylim(0.0, Tmax)
    ax.set_xlabel(r"$\alpha_k$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$T = 1/\beta_k$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    # panel titles "(a) k=2", ... are set in main()
    # ax.set_title(rf"Model C,  $k={k}$")
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
    path = os.path.join(outdir, f"modelC_lines_k{k}.csv")
    if k == 2:
        Ts = np.linspace(res["Tmin"], 1.5, 100)
        arr = np.column_stack([Ts, np.zeros_like(Ts), np.zeros_like(Ts), np.full_like(Ts, 1.0)])
        hdr = ("T, alpha_R(T) [= 0: zero capacity], alpha_M(T) [= 0], "
               "T_g(alpha=0) [retrieval only on the alpha = 0 axis, T < 1]")
        np.savetxt(path, arr, delimiter=",", header=hdr)
        return path
    curveR, curveG = res["curveR"], res["curveG"]
    Ts = np.unique(np.concatenate([curveR[:, 1], curveG[:, 1]]))
    aR = np.interp(Ts, curveR[:, 1], curveR[:, 0], left=curveR[0, 0], right=0.0)
    aG = np.interp(Ts, curveG[:, 1], curveG[:, 0], left=curveG[0, 0], right=0.0)
    aG[Ts > res["TM0"]] = 0.0
    Tg = np.array([Tg_line(a, k) for a in aR])
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
    lm = dict(k=k, Tc0_alpha0_exact=res["Tc0"], TM0_alpha0_exact=res["TM0"],
              alpha_c_T0_exact=res["ac0"], alpha_M_T0_exact=res["aM0"],
              alpha_c_Tmin=res["ac_low"], alpha_M_Tmin=res["alpha_M"],
              alpha_R_max=noseR[0], T_at_alpha_R_max=noseR[1],
              alpha_M_max=noseG[0], T_at_alpha_M_max=noseG[1],
              Tmin=res["Tmin"])
    path = os.path.join(outdir, f"modelC_landmarks_k{k}.csv")
    with open(path, "w") as f:
        f.write(f"# Model C landmarks at k = {k} (name, value)\n")
        for name, val in lm.items():
            f.write(f"{name},{val:.12g}\n")
    return path


# Self-test: closed forms and stationarity of the equations of state
def selftest():
    print("Model C self-test")
    print("  Kernel check M_k(1) = (2k-3)!!:", end=" ")
    for k in (4, 6, 8, 10, 12):
        Kernel(k)
    print("passed for k = 4, 6, 8, 10, 12.")

    print("  Closed forms vs Table II:")
    for k, Tc_ref, ac_ref in ((4, 0.25, 9.88e-3), (6, 0.148, 8.67e-5)):
        Tc = Tc0_alpha0(k)
        ac = alpha_c_zero_T(k)
        print(f"    k={k}:  Tc(0) = {Tc:.6g} (Table: {Tc_ref})   "
              f"alpha_c(0) = {ac:.6g} (Table: {ac_ref})   "
              f"alpha_M(0) = {alpha_M_zero_T(k):.6g}   "
              f"T_M(0) = {TM0_alpha0(k):.6g}")

    print("  Stationarity of beta*f at the solved branches "
          "(finite differences):")
    for k, alpha, T in ((4, 4e-3, 0.10), (6, 4e-5, 0.06), (10, 1e-9, 0.02)):
        kern = Kernel(k)
        ok, m, q = retrieval_solution(alpha, T, k, kern)
        assert ok, f"retrieval not found at k={k}, alpha={alpha}, T={T}"
        eps = 1e-7
        dfm = (free_energy(m + eps, q, alpha, T, k, kern)
               - free_energy(m - eps, q, alpha, T, k, kern)) / (2 * eps)
        dfq = (free_energy(m, q + eps, alpha, T, k, kern)
               - free_energy(m, q - eps, alpha, T, k, kern)) / (2 * eps)
        q0 = m0_solution(alpha, T, k, kern)
        dfq0 = (free_energy(0.0, q0 + eps, alpha, T, k, kern)
                - free_energy(0.0, q0 - eps, alpha, T, k, kern)) / (2 * eps)
        print(f"    k={k:2d} alpha={alpha:g} T={T}:  retrieval "
              f"(m={m:.4f}, q={q:.4f})  |df/dm|={abs(dfm):.2e}  "
              f"|df/dq|={abs(dfq):.2e};  m=0 branch (q={q0:.4f})  "
              f"|df/dq|={abs(dfq0):.2e}")

    print("  T = 0 capacity consistency: alpha_R(T=1e-4) vs Eq. (42):")
    for k in (4, 6, 10):
        kern = Kernel(k)
        aR = alpha_R_at_T(1e-4, k, kern)
        print(f"    k={k:2d}:  {aR:.6g}  vs  {alpha_c_zero_T(k):.6g}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="RS finite-temperature phase diagram of Model C "
                    "(class-H spherical memory).")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 6],
                    help="even hidden exponents k (default: 2 4 6)")
    ap.add_argument("--Tmin", type=float, default=0.01,
                    help="lowest temperature used as the T -> 0 proxy")
    ap.add_argument("--nT", type=int, default=160,
                    help="number of T grid points per k")
    ap.add_argument("--out", type=str, default=".",
                    help="output directory")
    ap.add_argument("--replot", action="store_true",
                    help="re-render the figure from the CSV files previously saved in --out (no numerics are re-run)")
    ap.add_argument("--selftest", action="store_true",
                    help="run self-test and exit")
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
        results = [compute_phase_diagram(k, Tmin=args.Tmin, nT=args.nT)
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
    # fig.suptitle("Model C (spherical visible spins): "
    #              "RS finite-temperature phase diagram", fontsize=12.5)
    tag = "_".join(f"k{k}" for k in args.k)
    png = os.path.join(args.out, f"modelC_phase_diagram_{tag}.png")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    print(f"figure  -> {png}")
    if args.replot:                        # in replot mode the CSVs are inputs
        return
    for res in results:
        print(f"csv     -> {save_csv(res, args.out)}")
        print(f"csv     -> {save_landmarks(res, args.out)}")


if __name__ == "__main__":
    main()
