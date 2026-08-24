"""
The MIT License (MIT) Copyright (c) 2026. Toshihiro Ota
"""

import argparse
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import brentq


# Unified style (shared with the Model A/C scripts)
COL_PARA, COL_RETR, COL_FROZ = "#bdd7ee", "#f8cfa4", "#c3e2ba"  # P, C, F
COL_SP, COL_AN = "#c1121f", "#7a0d16"                            # spinodal

# font sizes (enlarged for readability)
FS_TITLE, FS_LABEL, FS_TICK = 18, 17, 13
FS_LEGEND, FS_PHASE, FS_ANNOT, FS_RLAB = 14, 22, 13, 14


# Basic functions
def g(y):
    return y - 1.0 - np.log(y)


def eps_max(alpha):
    """Positive root of g(1+eps) = 2*alpha (extreme norm excess)."""
    if alpha <= 0.0:
        return 0.0
    f = lambda e: e - np.log1p(e) - 2.0 * alpha
    hi = 2.0 * alpha + 2.0 * np.sqrt(2.0 * alpha) + 10.0
    while f(hi) < 0.0:
        hi *= 2.0
    return brentq(f, 1e-14, hi, xtol=1e-12)


def kappa(t):
    """kappa(t) = (1/2)[t/(1-t) + log(1-t)], 0<t<1 (REM-type freezing rate)."""
    return 0.5 * (t / (1.0 - t) + np.log1p(-t))


def T_freeze(alpha):
    """CA-G freezing line T_f(alpha) = 1 + 1/eps_max(alpha)."""
    return 1.0 + 1.0 / eps_max(alpha)


def alpha_freeze(T):
    """Inverse of the freezing line: alpha_f(T) = g(1 + 1/(T-1))/2 for T > 1
    (NaN for T <= 1, where the freezing line does not exist)."""
    if T <= 1.0:
        return np.nan
    e = 1.0 / (T - 1.0)
    return 0.5 * (e - math.log1p(e))


# Equilibrium branches and their boundaries
def phi_P0(alpha, lam, T):
    """Paramagnetic branch phi_P; -inf if absent (lam >= 1)."""
    if lam >= 1.0:
        return -np.inf
    return alpha / (lam * T) - 0.5 * math.log(1.0 - lam)


def phi_C0(alpha, T):
    """Condensed sector max(phi_C, phi_F)."""
    beta = 1.0 / T
    pG = (1.0 + eps_max(alpha)) / (2.0 * T)
    if beta < 1.0 and alpha >= kappa(beta):
        pCA = alpha - 0.5 * math.log1p(-beta)
        return max(pCA, pG)
    return pG


def first_order_roots(lam, T, amax, ascan, EPS_scan):
    """All roots alpha of phi_P(alpha) = phi_C(alpha) at fixed T (lam < 1).

    ascan / EPS_scan: precomputed alpha grid and eps_max(ascan) used for
    bracketing; the roots themselves are refined with the exact branches.
    """
    if lam >= 1.0:
        return []
    beta = 1.0 / T
    pP = ascan / (lam * T) - 0.5 * math.log(1.0 - lam)
    pG = (1.0 + EPS_scan) / (2.0 * T)
    if beta < 1.0:
        pCA = np.where(ascan >= kappa(beta),
                       ascan - 0.5 * math.log1p(-beta), -np.inf)
        pC = np.maximum(pCA, pG)
    else:
        pC = pG
    dd = pP - pC
    d = lambda a: phi_P0(a, lam, T) - phi_C0(a, T)
    roots = []
    for i in np.where(np.sign(dd[:-1]) != np.sign(dd[1:]))[0]:
        try:
            roots.append(brentq(d, ascan[i], ascan[i + 1], xtol=1e-12))
        except ValueError:
            pass
    return roots


def triple_point(lam, amax):
    """Intersection of the freezing line with the P-condensed boundary
    (lam < 1): the P-C-F triple point."""
    def h(a):
        tf = T_freeze(a)
        return a / (lam * tf) - 0.5 * np.log(1.0 - lam) - (1.0 + eps_max(a)) / (2.0 * tf)
    aa = np.linspace(0.05, amax, 200)
    hh = np.array([h(a) for a in aa])
    idx = np.where(np.sign(hh[:-1]) != np.sign(hh[1:]))[0]
    if len(idx) == 0:
        return None
    i = idx[-1]                                      # main (largest-alpha) crossing
    a = brentq(h, aa[i], aa[i + 1])
    return a, T_freeze(a)


def alpha_th_zero_T(lam):
    """T = 0 intercept of the P-F equilibrium boundary:
    2*alpha/lam = 1 + eps_max(alpha); None if lam >= 1."""
    if lam >= 1.0:
        return None
    return brentq(lambda a: a / lam - (1.0 + eps_max(a)) / 2.0, 1e-3, 50.0)


# R spinodal: unified variational problem (annealed/frozen handled at once)
def Delta_one_quantum(alpha, lam, T, nx=240, ny=240):
    """One-quantum defection exponent Delta(alpha; lam, T).
    Delta < 0 <=> R locally stable."""
    if alpha <= 0.0:
        # limit in which the admissible set shrinks to the point (x,y)=(0,1)
        return -lam + lam * lam * T
    xm = np.sqrt(2.0 * alpha)
    yhi = 1.0 + eps_max(alpha)                       # upper root of g(y)=2*alpha
    ylo = brentq(lambda y: g(y) - 2.0 * alpha, 1e-12, 1.0, xtol=1e-14)
    x = np.linspace(-xm, xm, nx)
    y = np.linspace(ylo, yhi, ny)
    X, Y = np.meshgrid(x, y)
    I = 0.5 * X * X + 0.5 * (Y - 1.0 - np.log(Y))
    F = alpha - I - lam * (1.0 - X) + 0.5 * lam * lam * T * ((1.0 - X) ** 2 + Y)
    F = np.where(I <= alpha, F, -np.inf)
    return float(F.max())


def alpha_c_numeric(lam, T, ahi=1.2):
    """Root of Delta(alpha)=0 (Delta is increasing in alpha, so bisection)."""
    if Delta_one_quantum(1e-8, lam, T) >= 0.0:
        return 0.0
    if Delta_one_quantum(ahi, lam, T) <= 0.0:
        return np.nan
    lo, hi = 1e-8, ahi
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if Delta_one_quantum(mid, lam, T) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def alpha_c_annealed(lam, T):
    """Closed form of the annealed-leak spinodal."""
    lt = lam * lam * T
    if lt >= 1.0:
        return np.nan
    return lam * (2.0 - lam - lam * T) / (2.0 * (1.0 - lt)) + 0.5 * np.log(1.0 - lt)


def alpha_c_frozen(lam, T):
    """Closed form of the frozen-leak ceiling, small-T order."""
    z = 1.0 - 0.5 * lam * T
    return 0.5 * z * z if z > 0.0 else 0.0


def I_star(lam, T):
    """Existence rate of the annealed stationary target,
    I* = x*^2/2 + kappa(lam^2 T) (annealed / extreme-value switch).

    x* is the annealed defection saddle x* = lam(1 - lam T)/(1 - lam^2 T),
    the same saddle used by the numeric spinodal, the barrier,
    and the ladders; an earlier version of this function kept the
    superseded form x* = lam/(1 + lam T)."""
    lt = lam * lam * T
    if lt >= 1.0:
        return np.inf
    xs = lam * (1.0 - lam * T) / (1.0 - lam * lam * T)
    return 0.5 * xs * xs + kappa(lt)


# Optional overlays: escape barrier and nucleation ladders
def phi_R(p, alpha, lam, T):
    """Annealed retrieval-branch Landau curve:
    phi_R(p), A = 1 - lam(1-p)."""
    A = 1.0 - lam * (1.0 - p)
    return (1.0 - p) * alpha / (lam * T) - 0.5 * np.log(A) + p * p / (2.0 * T * A)


_ybounds_cache = {}


def _y_bounds(alpha):
    key = round(alpha, 12)
    if key not in _ybounds_cache:
        yhi = 1.0 + eps_max(alpha)
        ylo = brentq(lambda y: g(y) - 2.0 * alpha, 1e-12, 1.0, xtol=1e-14)
        _ybounds_cache[key] = (ylo, yhi)
    return _ybounds_cache[key]


def delta_sector(alpha, Lam, T, nb=240):
    """Unified sector exponent
    max_{I<=alpha}[alpha - I - Lam(1-x) + (Lam^2 T/2)((1-x)^2+y)].

    Same template as Delta_one_quantum (Lam = lam); with Lam = j*lam it gives
    the j-quantum CO-defection sector, whose endpoint j = n = 1/(lam*T) closes
    exactly onto the condensed value C/F (interior stationary point -> C,
    boundary -> F). Fast 1D implementation: interior candidate in closed
    form + boundary scan I = alpha.
    """
    if alpha <= 0.0:
        return -Lam + Lam * Lam * T
    lt = Lam * Lam * T
    best = -np.inf
    if lt < 1.0:                                      # interior stationary point
        xs = Lam * (1.0 - Lam * T) / (1.0 - lt)
        ys = 1.0 / (1.0 - lt)
        Ii = 0.5 * xs * xs + 0.5 * g(ys)
        if Ii <= alpha:
            best = alpha - Ii - Lam * (1.0 - xs) + 0.5 * lt * ((1.0 - xs) ** 2 + ys)
    ylo, yhi = _y_bounds(alpha)                       # boundary I = alpha
    y = np.linspace(ylo, yhi, nb)
    xb = np.sqrt(np.clip(2.0 * alpha - (y - 1.0 - np.log(y)), 0.0, None))
    for s in (1.0, -1.0):
        Fb = -Lam * (1.0 - s * xb) + 0.5 * lt * ((1.0 - s * xb) ** 2 + y)
        best = max(best, float(np.max(Fb)))
    return best


def _kmax_distinct(lam, T):
    """Largest admissible rung on the distinct ladder (A_k = 1 - k lam^2 T > 0)."""
    n = 1.0 / (lam * T)
    if lam < 1.0:
        return int(np.floor(n))
    return int(np.floor(min(n, (1.0 - 1e-9) / (lam * lam * T))))


def barrier_height(alpha, lam, T):
    """Escape barrier of R (rate units): Delta phi_b = beta/2 - min phi along
    the lower of the two quantised ladders (RS / one-quantum, leading order).

    The distinct-defection ladder counts as an escape path only for lam < 1
    (annealed rung values phi_R(p_k)); for lam >= 1 it cannot reach P
    (A_k = 1 - k lam^2 T > 0 fails before k = n), so R escapes via the
    co-defection ladder alone."""
    n = 1.0 / (lam * T)
    if n < 1.0 or alpha <= 0.0:
        return np.nan
    half = 0.5 / T
    cands = []
    kmax = _kmax_distinct(lam, T)                     # distinct ladder, annealed rungs
    if lam < 1.0 and kmax >= 1:
        ks = np.arange(1, kmax + 1)
        cands.append(half - float(np.min(phi_R(1.0 - ks * lam * T, alpha, lam, T))))
    jint = list(range(1, min(int(np.floor(n)), 12) + 1))   # co ladder, unified sectors
    js = jint + [j for j in np.linspace(jint[-1], n, 5)[1:]] + [n]
    cands.append(-min(delta_sector(alpha, j * lam, T) for j in js))
    return max(min(cands), 0.0)


def _clabel_positions(cs, avoid, xspan, yspan, minscore=0.05, edge=0.03):
    """One label position per contour level: the contour vertex farthest
    (min-distance in axes-fraction units) from the overlaid curves, the axes
    edges, and the labels already placed; levels with no sufficiently clear
    vertex are left unlabeled. Returns data-coordinate positions. Required
    clearances: `edge` from the axes box, `minscore` from the curves, and
    `minscore` + 0.05 between labels."""
    pos = []
    pts = list(avoid)                                  # normalized polylines
    lab = []                                           # placed labels
    for segs in cs.allsegs:
        best, bestscore = None, minscore
        for seg in segs:
            if len(seg) < 5:
                continue
            xn, yn = seg[:, 0] / xspan, seg[:, 1] / yspan
            score = (np.minimum.reduce([xn, 1.0 - xn, yn, 1.0 - yn]) + (minscore - edge))
            if pts:
                A = np.vstack(pts)
                d2 = ((xn[:, None] - A[None, :, 0]) ** 2
                      + (yn[:, None] - A[None, :, 1]) ** 2)
                score = np.minimum(score, np.sqrt(d2.min(axis=1)))
            if lab:
                L = np.vstack(lab)
                d2 = ((xn[:, None] - L[None, :, 0]) ** 2
                      + (yn[:, None] - L[None, :, 1]) ** 2)
                # labels repel each other more strongly than curves do
                score = np.minimum(score, np.sqrt(d2.min(axis=1)) - 0.05)
            i = int(np.argmax(score))
            if score[i] > bestscore:
                best, bestscore = (float(seg[i, 0]), float(seg[i, 1])), score[i]
        if best is not None:
            pos.append(best)
            lab.append(np.array([[best[0] / xspan, best[1] / yspan]]))
    return pos


def overlay_barrier(ax, lam, res, Tmax, nb=76,
                    levels=(0.05, 0.1, 0.2, 0.5, 1.0), avoid_pts=()):
    """Contours of Delta phi_b inside the metastable R region.

    avoid_pts: extra normalized points (e.g. the ladder-inset reference
    star) that the contour labels must keep clear of."""
    Ts_sp, ac = res["Ts_sp"], res["ac"]
    amax_R = float(np.nanmax(ac))
    if amax_R <= 1e-3:
        return
    aB = np.linspace(1e-3, amax_R * 1.02, nb)
    TB = np.linspace(0.02, Tmax, nb)
    ABg, TTBg = np.meshgrid(aB, TB)
    acT = np.interp(TB, Ts_sp, ac)
    Bar = np.full_like(ABg, np.nan)
    for i, t in enumerate(TB):
        for jx, a in enumerate(aB):
            if a < acT[i]:
                Bar[i, jx] = barrier_height(a, lam, t)
    cs = ax.contour(ABg, TTBg, Bar, levels=list(levels),
                    colors="navy", linewidths=1.0, linestyles="--")
    # collision-free label positions (kept clear of the spinodal, the dashed asymptotics, the I* line, the axes edges, and each other)
    xspan, yspan = res["amax"], Tmax
    curves = [ac, res["an"] if lam < 1.0 else res["fz"]]
    if lam < 1.0:
        curves.append(res["istar"])
    avoid = list(avoid_pts)
    for c in curves:
        c = np.asarray(c, float)
        m = np.isfinite(c) & (c > 0.0)
        if m.any():
            avoid.append(np.column_stack([c[m] / xspan, Ts_sp[m] / yspan]))
    pos = _clabel_positions(cs, avoid, xspan, yspan)
    if pos:
        ax.clabel(cs, fmt="%g", fontsize=10, inline=True, manual=pos)


def draw_ladder_inset(ax, lam, alpha0, T0):
    """Inset: phi along the two escape ladders at the reference point (alpha0, T0).

    The two ladder-line legend entries are merged into the figure-level
    legend (see legend_handles, labels prefixed 'inset:')."""
    beta = 1.0 / T0
    n = 1.0 / (lam * T0)
    # overlap with panel content is sanctioned: the inset covers roughly the upper-right quarter of each panel (same rect for both regimes)
    rect = [0.51, 0.52, 0.47, 0.46]
    axi = ax.inset_axes(rect)
    axi.axhline(beta / 2.0, color="0.45", lw=0.8, ls="--")
    axi.plot([0.0], [beta / 2.0], "o", color="0.2", ms=4)
    axi.text(0.015, beta / 2.0, " R", fontsize=10, va="bottom")

    finite_vals = [beta / 2.0]

    # distinct-defection ladder -> P (continuous annealed curve + quantised rungs)
    wmax_d = min(1.0, 1.0 / lam)
    w = np.linspace(1e-4, wmax_d * (1.0 - 1e-6), 300)
    with np.errstate(divide="ignore", invalid="ignore"):
        axi.plot(w, phi_R(1.0 - w, alpha0, lam, T0), color="tab:blue", lw=1.3,
                 label=r"distinct ladder $\to$ P")
    kmax = _kmax_distinct(lam, T0)
    if lam < 1.0:
        if kmax >= 1:
            ks = np.arange(1, kmax + 1)
            pk = phi_R(1.0 - ks * lam * T0, alpha0, lam, T0)
            axi.plot(ks * lam * T0, pk, "o", color="tab:blue", ms=3.5)
            finite_vals += list(pk)
        pP = phi_R(0.0, alpha0, lam, T0)
        axi.plot([1.0], [pP], "s", color="tab:blue", ms=4.5)
        axi.text(1.0, pP, "P ", fontsize=10, ha="right", va="bottom")
        finite_vals.append(pP)
    else:
        axi.text(0.985, beta / 2.0, "distinct: blocked ($A_k>0$ fails) ",
                 fontsize=9, color="tab:blue", ha="right", va="top")

    # co-defection ladder -> C/F (unified sectors, exact incl. frozen endpoint)
    js = np.linspace(1.0, n, 160)
    pc = np.array([beta / 2.0 + delta_sector(alpha0, j * lam, T0) for j in js])
    axi.plot(js * lam * T0, pc, color="tab:green", lw=1.3,
             label=r"co ladder $\to$ C/F")
    jd = np.arange(1, int(np.floor(n)) + 1)
    pd = np.array([beta / 2.0 + delta_sector(alpha0, j * lam, T0) for j in jd])
    axi.plot(jd * lam * T0, pd, "o", color="tab:green", ms=3.5)
    pend = beta / 2.0 + delta_sector(alpha0, n * lam, T0)
    axi.plot([1.0], [pend], "^", color="tab:green", ms=5)
    axi.text(1.0, pend, "C/F ", fontsize=10, ha="right", va="bottom")
    finite_vals += list(pc) + [pend]

    lo, hi = min(finite_vals), max(finite_vals)
    pad = 0.12 * (hi - lo + 1e-9)
    axi.set_xlim(0.0, 1.05)
    axi.set_ylim(lo - pad, hi + pad)
    axi.set_xlabel(r"transferred weight $1-p$", fontsize=10, labelpad=2)
    axi.set_ylabel(r"$\varphi$", fontsize=10, labelpad=1)
    axi.tick_params(labelsize=9)
    # the ladder-line entries are listed in the figure legend ('inset:' ...) instead of a per-inset legend
    # title moved inside the inset
    # axi.set_title(rf"escape ladders at $(\alpha,T)=({alpha0:.2f},{T0:.2f})$", fontsize=7)
    axi.text(0.03, 0.97, "escape ladders at\n"
             + rf"$(\alpha,T)=({alpha0:.2f},{T0:.2f})$",
             transform=axi.transAxes, fontsize=9, ha="left", va="top")
    ax.plot([alpha0], [T0], marker="*", ms=16.5, mfc="gold", mec="k", lw=0)


# Per-lambda pipeline: compute_phase_diagram(...) -> dict
def compute_grid(lam, amax, Tmax, nA=460, nT=460):
    """Equilibrium branches on the (alpha, T) grid.

    Closed forms only (cheap); re-evaluated also in --replot mode, in which
    the expensive line computations are read back from the CSV files."""
    alphas = np.linspace(1e-4, amax, nA)
    Ts = np.linspace(0.02, Tmax, nT)
    EPS = np.array([eps_max(a) for a in alphas])

    A, TT = np.meshgrid(alphas, Ts)
    EPSg = np.broadcast_to(EPS, A.shape)
    beta = 1.0 / TT

    if lam < 1.0:
        phiP = A / (lam * TT) - 0.5 * np.log(1.0 - lam)
    else:
        phiP = np.full_like(A, -np.inf)
    phiG = (1.0 + EPSg) / (2.0 * TT)
    b = np.minimum(beta, 1.0 - 1e-12)
    kap = np.where(beta < 1.0, 0.5 * (b / (1.0 - b) + np.log1p(-b)), np.inf)
    phiCA = np.where((beta < 1.0) & (A >= kap), A - 0.5 * np.log1p(-b), -np.inf)
    phiC = np.maximum(phiCA, phiG)
    phase = np.where(phiP > phiC, 0.0, np.where(phiCA > phiG, 1.0, 2.0))
    return dict(alphas=alphas, Ts=Ts, EPS=EPS, A=A, TT=TT,
                phase=phase, phiP=phiP, phiC=phiC)


def compute_phase_diagram(lam, amax, Tmax, nA=460, nT=460, nTs=90,
                          verbose=True):
    """All curves and landmarks of one (alpha, T) panel at fixed lambda."""
    t0 = time.time()
    grid = compute_grid(lam, amax, Tmax, nA, nT)
    alphas, EPS = grid["alphas"], grid["EPS"]

    # --- common T grid for all exported lines ---
    Ts_sp = np.linspace(0.02, Tmax, nTs)

    # first-order P-condensed boundary alpha_PC(T): root-refined values,
    # up to NPC roots per T, sorted ascending (NaN-padded). Two roots occur
    # for -1/log(1-lam) < T < 1/lam, where the P phase re-enters at small
    # alpha (phi_P(0) = -log(1-lam)/2 exceeds phi_G(0) = 1/(2T)).
    NPC = 3
    aPC = np.full((nTs, NPC), np.nan)
    if lam < 1.0:
        for i, t in enumerate(Ts_sp):
            roots = sorted(first_order_roots(lam, t, amax, alphas, EPS))
            aPC[i, :min(len(roots), NPC)] = roots[:NPC]

    # C-F freezing line alpha_f(T) (closed form). It is the equilibrium
    # C-F boundary where the condensed sector is the equilibrium at alpha_f,
    # i.e. where phi_C(alpha_f) >= phi_P(alpha_f).
    aF = np.array([alpha_freeze(t) for t in Ts_sp])
    eqF = np.isfinite(aF)
    if lam < 1.0:
        for i, t in enumerate(Ts_sp):
            if eqF[i]:
                eqF[i] = phi_C0(aF[i], t) >= phi_P0(aF[i], lam, t)

    # R spinodal (numeric, unified variational) and closed-form asymptotics
    ac = np.array([alpha_c_numeric(lam, t) for t in Ts_sp])
    an = np.array([alpha_c_annealed(lam, t) for t in Ts_sp])
    fz = np.array([alpha_c_frozen(lam, t) for t in Ts_sp])
    istar = np.array([I_star(lam, t) for t in Ts_sp])

    # --- landmarks ---
    lm = {
        "lam": lam, "amax": amax, "Tmax": Tmax,
        "alpha_c_T0_theory": lam - 0.5 * lam * lam if lam < 1.0 else 0.5,
        "alpha_c_numeric_Tmin": float(ac[0]),
        "T_endpoint_R": 1.0 / lam,          # n = beta/lam = 1
    }
    if lam < 1.0:
        tp = triple_point(lam, max(amax, 10.0))
        if tp is not None:
            lm["triple_point_alpha"], lm["triple_point_T"] = tp
        ath = alpha_th_zero_T(lam)
        if ath is not None:
            lm["alpha_th_T0_PG"] = ath
        # alpha = 0 intercept of the P-F boundary: phi_P(0) = phi_F(0)
        lm["T_P_reentrant_alpha0"] = -1.0 / math.log(1.0 - lam)
    for t in (0.02, 0.4, 1.0):              # validation values (numeric vs closed)
        lm[f"alpha_c_numeric_T{t:g}"] = alpha_c_numeric(lam, t)
        lm[f"alpha_c_annealed_T{t:g}"] = alpha_c_annealed(lam, t)
        lm[f"alpha_c_frozen_T{t:g}"] = alpha_c_frozen(lam, t)

    res = dict(lam=lam, amax=amax, Tmax=Tmax, **grid,
               Ts_sp=Ts_sp, ac=ac, an=an, fz=fz, istar=istar,
               aPC=aPC, aF=aF, eqF=eqF, landmarks=lm)
    if verbose:
        print(f"[lam={lam:g}]  numeric spinodal vs closed forms "
              f"(62)/(63):")
        print("      T     numeric   (62) ann   (63) froz")
        for t in (0.02, 0.4, 1.0):
            print(f"    {t:5.2f}   {lm[f'alpha_c_numeric_T{t:g}']:7.4f}   "
                  f"{lm[f'alpha_c_annealed_T{t:g}']:8.4f}   "
                  f"{lm[f'alpha_c_frozen_T{t:g}']:8.4f}")
        print(f"    theory T=0 intercept: {lm['alpha_c_T0_theory']:.4f}   "
              f"R endpoint T = 1/lam = {lm['T_endpoint_R']:.4f}")
        if "triple_point_alpha" in lm:
            print(f"    triple point: alpha* = {lm['triple_point_alpha']:.3f}, "
                  f"T* = {lm['triple_point_T']:.3f}")
        if "alpha_th_T0_PG" in lm:
            print(f"    P-G boundary T=0 intercept: "
                  f"alpha_th(0) = {lm['alpha_th_T0_PG']:.3f}")
        print(f"    [{time.time() - t0:.1f} s]")
    return res


def load_phase_diagram(lam, outdir, nA=460, nT=460, verbose=True):
    """Rebuild the result dict from previously saved CSV files (--replot).

    No expensive numerics are re-run: all boundary lines and landmarks are
    read back from modelB_lines_lam*.csv / modelB_landmarks_lam*.csv; only
    the closed-form equilibrium branches on the fill grid are re-evaluated.
    """
    lmpath = os.path.join(outdir, f"modelB_landmarks_lam{lam:g}.csv")
    lnpath = os.path.join(outdir, f"modelB_lines_lam{lam:g}.csv")
    lm = {}
    with open(lmpath) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, val = line.strip().split(",")
            lm[name] = float(val)
    d = np.genfromtxt(lnpath, delimiter=",")
    amax, Tmax = lm["amax"], lm["Tmax"]
    grid = compute_grid(lam, amax, Tmax, nA, nT)
    # I* is a closed form; re-evaluate it here so a corrected I_star supersedes CSV column 4
    istar = np.array([I_star(lam, t) for t in d[:, 0]])
    res = dict(lam=lam, amax=amax, Tmax=Tmax, **grid,
               Ts_sp=d[:, 0], ac=d[:, 1], an=d[:, 2], fz=d[:, 3],
               istar=istar, aPC=d[:, 5:8], aF=d[:, 8],
               eqF=d[:, 9] > 0.5, landmarks=lm)
    if verbose:
        print(f"[lam={lam:g}]  replotted from {lnpath}")
    return res


def plot_panel(ax, res):
    lam, amax, Tmax = res["lam"], res["amax"], res["Tmax"]
    A, TT, phase = res["A"], res["TT"], res["phase"]
    Ts_sp = res["Ts_sp"]

    # equilibrium phase fills: P (paramagnetic), C (condensed), F (frozen)
    ax.contourf(A, TT, phase, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=[COL_PARA, COL_RETR, COL_FROZ])

    # first-order transition line P-condensed (black solid; all branches,
    # drawn as the zero contour of phi_P - phi_C on the grid)
    if lam < 1.0:
        ax.contour(A, TT, res["phiP"] - res["phiC"], levels=[0.0],
                   colors="k", linewidths=2.0)

    # freezing line C-F (continuous; black dotted, equilibrium part only)
    m = res["eqF"]
    ax.plot(res["aF"][m], Ts_sp[m], ls=":", color="k", lw=2.0)

    # R spinodal (numeric, unified variational) and metastable region (hatched)
    ac = res["ac"]
    ax.fill_betweenx(Ts_sp, 0.0, ac, facecolor="none", hatch="////", edgecolor=COL_SP, linewidth=0.0)
    ax.plot(ac, Ts_sp, color=COL_SP, lw=2.4)

    # asymptotic closed forms (dashed, for comparison)
    if lam < 1.0:
        an = res["an"]
        m = np.isfinite(an) & (an > 0.0)
        ax.plot(an[m], Ts_sp[m], ls="--", color=COL_AN, lw=1.3)
        istar = res["istar"]
        m2 = np.isfinite(istar) & (istar < ac)        # inside the R region only
        ax.plot(istar[m2], Ts_sp[m2], ls="-.", color="0.35", lw=1.1)
    else:
        fz = res["fz"]
        m = fz > 0.0
        ax.plot(fz[m], Ts_sp[m], ls="--", color=COL_AN, lw=1.3)

    ax.set_xlim(0.0, amax)
    ax.set_ylim(0.0, Tmax)
    ax.set_xlabel(r"$\alpha$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$T$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)


def label_regions(ax, res):
    """Placement of phase labels (region medians) and the R label."""
    A, TT, phase = res["A"], res["TT"], res["phase"]
    lam, amax, Tmax = res["lam"], res["amax"], res["Tmax"]
    for idx, name in ((0.0, "P"), (1.0, "C"), (2.0, "F")):
        msk = phase == idx
        if msk.sum() < 50:
            continue
        ax.text(np.median(A[msk]), np.median(TT[msk]), name,
                fontsize=FS_PHASE, weight="bold", ha="center", va="center")
    ac = res["ac"]
    if np.nanmax(ac) > 0.02:
        # place the R label just to the RIGHT of the spinodal (and of the
        # dashed closed-form asymptotics), at a height clear of the other
        # labels: low in the panel for lam < 1, above the spinodal knee
        # (but below T = 1/lam) for lam >= 1
        if lam < 1.0:
            y0, dash, xoff = 0.20 * Tmax, res["an"], 0.03 * amax
        else:
            y0, dash, xoff = 0.62 * min(1.0 / lam, Tmax), res["fz"], 0.10 * amax
        x0 = float(np.interp(y0, res["Ts_sp"], ac))
        xd = float(np.interp(y0, res["Ts_sp"], dash))
        if np.isfinite(xd):
            x0 = max(x0, xd)
        ax.text(x0 + xoff, y0, "R metastable\n(typical retrieval)",
                fontsize=FS_RLAB, color=COL_SP, ha="left", va="center")


def annotate_special(ax, res, overlay=False):
    """T=0 intercepts, triple point, and the high-T endpoint T = 1/lam of R.

    overlay=True (barrier/paths figure): on the lam < 1 panel the ladder
    inset covers the upper-right quarter, so the 'triple point' and
    'T = 1/lam' annotations that live there are suppressed."""
    lam, amax, Tmax = res["lam"], res["amax"], res["Tmax"]
    lm = res["landmarks"]
    hide_ur = overlay and lam < 1.0
    # exact T = 0 intercept of the spinodal: lam - lam^2/2 (lam < 1), 1/2 (lam >= 1);
    # the lowest-T grid value res["ac"][0] is only its T = Tmin proxy and is not quoted in the figure
    a0 = lm["alpha_c_T0_theory"]
    if a0 > 1e-3:                                    # spinodal T=0 intercept
        ax.annotate(rf"$\alpha_c(0)={a0:g}$", xy=(a0, 0.0),
                    xytext=(a0 + 0.10 * amax, 0.06 * Tmax), fontsize=FS_ANNOT,
                    color=COL_SP,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=COL_SP))
    if 1.0 / lam <= Tmax and not hide_ur:            # endpoint of R (n = beta/lam = 1)
        if lam < 1.0:
            # text in the empty P region, upper part of the panel
            xyt, ha = (0.42 * amax, 0.90 * Tmax), "left"
        else:
            # text up-left of the F label, centred midway between the
            # T axis and the F label (placed at the F-region median)
            mskF = res["phase"] == 2.0
            xF = float(np.median(res["A"][mskF]))
            yF = float(np.median(res["TT"][mskF]))
            xyt, ha = (0.5 * xF, yF + 0.17 * Tmax), "center"
        ax.annotate(r"$T=1/\lambda$  ($n=\beta/\lambda=1$)",
                    xy=(0.004 * amax, 1.0 / lam), xytext=xyt, ha=ha,
                    fontsize=FS_ANNOT, color=COL_SP,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=COL_SP))
    if "alpha_th_T0_PG" in lm and lm["alpha_th_T0_PG"] < amax:
        ath = lm["alpha_th_T0_PG"]                   # P-F boundary intercept at T=0
        ax.annotate(rf"$\alpha_{{\rm th}}(0)\simeq{ath:.2f}$", xy=(ath, 0.0),
                    xytext=(ath + 0.05 * amax, 0.07 * Tmax), fontsize=FS_ANNOT,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    if ("triple_point_alpha" in lm and lm["triple_point_T"] <= Tmax
            and not hide_ur):
        tp = (lm["triple_point_alpha"], lm["triple_point_T"])
        ax.plot([tp[0]], [tp[1]], "ko", ms=5)
        ax.annotate(f"triple point\n({tp[0]:.2f}, {tp[1]:.2f})", xy=tp,
                    xytext=(tp[0] + 0.09 * amax, tp[1] + 0.03 * Tmax),
                    fontsize=FS_ANNOT,
                    arrowprops=dict(arrowstyle="-", lw=0.7))


def legend_handles(barrier=False, paths=False):
    handles = [
        Patch(fc=COL_PARA, label="P (paramagnetic)"),
        Patch(fc=COL_RETR, label="C (condensed; mixture of retrieval lumps)"),
        Patch(fc=COL_FROZ, label="F (frozen; max-norm retrieval)"),
        Patch(fc="none", hatch="////", ec=COL_SP, label="R metastable region (typical retrieval)"),
        Line2D([], [], color="k", lw=2.0, label="first-order transition (equilibrium)"),
        Line2D([], [], color=COL_SP, lw=2.4,
               label=r"spinodal $\alpha_c(\lambda,T)$ (unified variational, numeric)"),
        Line2D([], [], color="k", lw=2.0, ls=":",
               label=r"freezing line $T_f(\alpha)$ (continuous, C$-$F)"),
        Line2D([], [], color=COL_AN, lw=1.3, ls="--",
               label="closed-form asymptotics"),
        Line2D([], [], color="0.35", lw=1.1, ls="-.",
               label=r"leak-freezing line $I^*(T)$ (annealed / extreme-value switch)"),
    ]
    if barrier:
        handles.append(Line2D([], [], color="navy", lw=1.0, ls="--",
                              label=r"barrier contours $\Delta\varphi_b$"
                                    r" (levels $0.05,0.1,0.2,0.5,1$)"))
    if paths:
        handles.append(Line2D([], [], marker="*", color="none", mfc="gold",
                              mec="k", ms=10, label="ladder-inset reference point"))
        # legend of the ladder inset, merged into the figure legend; the
        # 'inset:' prefix marks entries that refer to the inset curves
        handles.append(Line2D([], [], color="tab:blue", lw=1.3,
                              label=r"inset: distinct ladder $\to$ P"))
        handles.append(Line2D([], [], color="tab:green", lw=1.3,
                              label=r"inset: co ladder $\to$ C/F"))
    return handles


def add_fig_legend(fig, handles, fontsize=FS_LEGEND, max_ncol=3):
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
    """All boundary lines of one panel on the common T grid Ts_sp."""
    lam = res["lam"]
    path = os.path.join(outdir, f"modelB_lines_lam{lam:g}.csv")
    hdr = (f"Model B boundary lines at lambda = {lam:g} "
           "(exponential load N_h = e^{alpha N_v}, Gaussian patterns).\n"
           "All lines are graphs alpha(T) on a common T grid; NaN = absent.\n"
           "alpha_R_spinodal: numeric root of the unified one-quantum "
           "variational problem;\n"
           "alpha_R_annealed / alpha_R_frozen: closed forms, Eqs. (62)/(63);\n"
           "I_star: leak-freezing line (annealed / extreme-value switch);\n"
           "alpha_PC_1..3: roots of phi_P = phi_C at fixed T, sorted "
           "ascending (P-condensed first-order boundary; lam < 1 only; "
           "two roots where the P phase re-enters at small alpha, "
           "-1/log(1-lam) < T < 1/lam);\n"
           "alpha_CAG_freezing: T_f(alpha) = 1 + 1/eps_max(alpha) inverted, "
           "alpha_f(T) = g(1 + 1/(T-1))/2; it is the CA-G equilibrium "
           "boundary only where freezing_is_equilibrium = 1.\n"
           "T, alpha_R_spinodal, alpha_R_annealed, alpha_R_frozen, I_star, "
           "alpha_PC_1, alpha_PC_2, alpha_PC_3, alpha_CAG_freezing, "
           "freezing_is_equilibrium")
    arr = np.column_stack([res["Ts_sp"], res["ac"], res["an"], res["fz"],
                           res["istar"], res["aPC"], res["aF"],
                           res["eqF"].astype(float)])
    np.savetxt(path, arr, delimiter=",", header=hdr)
    return path


def save_landmarks(res, outdir):
    """Special points / validation values of one panel (name, value rows)."""
    lam = res["lam"]
    path = os.path.join(outdir, f"modelB_landmarks_lam{lam:g}.csv")
    with open(path, "w") as f:
        f.write(f"# Model B landmarks at lambda = {lam:g} (name, value)\n")
        for name, val in res["landmarks"].items():
            f.write(f"{name},{val:.12g}\n")
    return path


def selftest():
    print("Model B self-test")
    # eps_max / freezing-line inverses
    for a in (0.05, 0.3, 1.0):
        e = eps_max(a)
        assert abs(g(1.0 + e) - 2.0 * a) < 1e-9
        tf = T_freeze(a)
        assert abs(alpha_freeze(tf) - a) < 1e-9
    print("  eps_max root and alpha_freeze = T_freeze^{-1}: passed.")
    # numeric spinodal vs annealed closed form in the annealed regime
    for lam, T in ((0.5, 0.02), (0.5, 0.4), (0.5, 1.0)):
        nm, an = alpha_c_numeric(lam, T), alpha_c_annealed(lam, T)
        print(f"  spinodal lam={lam} T={T}: numeric {nm:.4f} vs "
              f"annealed (62) {an:.4f}")
        assert abs(nm - an) < 2e-3
    # T -> 0 intercepts vs Eq. (55)
    for lam in (0.5, 1.5):
        nm = alpha_c_numeric(lam, 0.005)
        th = lam - 0.5 * lam * lam if lam < 1.0 else 0.5
        print(f"  T->0 capacity lam={lam}: numeric {nm:.4f} vs "
              f"Eq. (55) {th:.4f}")
        assert abs(nm - th) < 5e-3
    # fast sector template vs 2D-grid evaluator
    d1 = Delta_one_quantum(0.19, 0.5, 1.0)
    d2 = delta_sector(0.19, 0.5, 1.0)
    print(f"  delta_sector vs Delta_one_quantum at "
          f"(alpha,lam,T)=(0.19,0.5,1.0): {d2:.5f} vs {d1:.5f}")
    assert abs(d1 - d2) < 1e-4
    print("  all checks passed.")


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Finite-temperature phase diagram of Model B "
                    "(class-H modern Hopfield network).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lam", type=float, nargs="+", default=[0.5, 1.5],
                   help="softmax sharpness lambda, one value per panel")
    p.add_argument("--amax", type=float, nargs="+", default=None,
                   help="alpha-axis maximum per panel (default: 2.0 if lam<1 else 1.0; one value applies to all)")
    p.add_argument("--Tmax", type=float, default=2.0,
                   help="temperature-axis maximum (all panels)")
    p.add_argument("--barrier", action="store_true",
                   help="overlay contours of the escape barrier Delta phi_b (RS / one-quantum, leading order)")
    p.add_argument("--paths", action="store_true",
                   help="overlay the two nucleation ladders (distinct-defection -> P, co-defection -> C/F) as an inset at a reference point")
    p.add_argument("--path-point", type=float, nargs=2, metavar=("ALPHA", "T"), default=None,
                   help="reference point for the ladder inset (default: auto, inside the R region of each panel)")
    p.add_argument("--out", type=str, default=".",
                   help="output directory")
    p.add_argument("--replot", action="store_true",
                   help="re-render the figure from the CSV files previously saved in --out (skips the numerical line computations; --amax/--Tmax are read from the landmarks file)")
    p.add_argument("--selftest", action="store_true",
                   help="run self-test and exit")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return

    lams = args.lam
    if args.amax is None:
        amaxs = [2.0 if l < 1.0 else 1.0 for l in lams]
    elif len(args.amax) == 1:
        amaxs = args.amax * len(lams)
    else:
        if len(args.amax) != len(lams):
            raise SystemExit("--amax must have 1 value or as many values as --lam")
        amaxs = args.amax

    os.makedirs(args.out, exist_ok=True)

    if args.replot:
        results = [load_phase_diagram(lam, args.out) for lam in lams]
    else:
        results = [compute_phase_diagram(lam, amax, args.Tmax)
                   for lam, amax in zip(lams, amaxs)]

    npan = len(results)
    fig, axes = plt.subplots(1, npan, figsize=(6.2 * npan, 5.4),
                             constrained_layout=True, squeeze=False)
    for i, (ax, res) in enumerate(zip(axes[0], results)):
        lam = res["lam"]
        plot_panel(ax, res)
        ax.set_title(rf"({chr(97 + i)})  $\lambda = {lam:g}$  "
                     + ("($\\lambda<1$: P phase present)" if lam < 1.0
                        else "($\\lambda\\geq 1$: no P phase)"),
                     fontsize=FS_TITLE)
        label_regions(ax, res)
        annotate_special(ax, res, overlay=(args.barrier or args.paths))
        a0 = T0 = None
        if args.paths:
            if args.path_point is not None:
                a0, T0 = args.path_point
            else:
                T0 = 0.4 / max(lam, 1.0)
                # reference point from the exported spinodal line (no numerics)
                a0 = 0.5 * float(np.interp(T0, res["Ts_sp"], res["ac"]))
        if args.barrier:
            pts = ([np.array([[a0 / res["amax"], T0 / res["Tmax"]]])]
                   if a0 is not None else [])
            overlay_barrier(ax, lam, res, res["Tmax"], avoid_pts=pts)
        if args.paths:
            if a0 > 1e-3:
                draw_ladder_inset(ax, lam, a0, T0)
            else:
                print(f"[warn] panel {i}: no R region at the reference point; "
                      f"ladder inset skipped")

    add_fig_legend(fig, legend_handles(args.barrier, args.paths))
    # fig.suptitle(r"Model B (attention / modern Hopfield): finite-temperature phase diagram (exponential load $N_h=e^{\alpha N_v}$, Gaussian patterns")"
    #               + "  \u2014  real-temperature construction", fontsize=12.5)

    tag = "_".join(f"lam{l:g}" for l in lams)
    if args.barrier or args.paths:         # keep the plain figure distinct
        tag += "_full"
    png = os.path.join(args.out, f"modelB_phase_diagram_{tag}.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"figure  -> {png}")
    if not args.replot:                    # in replot mode the CSVs are inputs
        for res in results:
            print(f"csv     -> {save_csv(res, args.out)}")
            print(f"csv     -> {save_landmarks(res, args.out)}")


if __name__ == "__main__":
    main()
