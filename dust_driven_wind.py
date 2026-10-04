#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Toy Model for Dust-driven Winds
=================================
Theoretical Astrophysics, HT 2026 -- Assignment (S. Hoefner)

A single fluid element is followed from the photosphere (r0 = R_*), where a
shock gives it an initial velocity u0, through the dust-free zone (Gamma = 0)
into the dusty wind (Gamma = const) beyond the condensation distance R_c.

Equation of motion (Eq. 1 of the assignment):

    du/dt = -g0 (r0/r)^2 (1 - Gamma),      u = dr/dt,     g0 = G M_* / r0^2

Dimensionless form (r' = r/R_*, u' = u/u_esc, t' = t/t_ff, with
u_esc = sqrt(2 G M_* / R_*) and t_ff = R_* / u_esc), as a first-order system:

    dr'/dt' = u'
    du'/dt' = -(1 - Gamma) / (2 r'^2)

Analytical solution for constant Gamma (Eq. 4 of the assignment):

    u'(r2)^2 - u'(r1)^2 = (1 - Gamma) (1/r2' - 1/r1')

Tasks covered
-------------
 0. Code tests (analytical free-fall solution, constant-velocity case,
    consistency with the lecture notes).
 1+2. Integration and reproduction of Fig. 1 of the assignment.
 3. Gamma and R_c from the amorphous-carbon grain properties (Table 1) and
    comparison with the test cases.
 4. Velocity vs. distance (km/s, units of R_*), numerical-vs-analytical
    comparison table, terminal velocity.
 5. Velocity vs. distance in AU, planetary orbits, escape velocity, Pluto.

Usage
-----
    python dust_driven_wind.py [--outdir DIR] [--dpi N]

Requirements: numpy, scipy, matplotlib. All figures, a CSV table and a text
summary are written to the output directory (default: ./wind_output).

Modelling choices (all stated here so they can be quoted in the report)
-----------------------------------------------------------------------
* t_ff := R_* / u_esc(R_*). With this scale, the Gamma = 0 trajectory peaks at
  ~1.59 yr, as in Fig. 1 of the assignment. (FreeFall.py was not available when
  this script was written; if your course uses another definition of t_ff,
  change `Star.t_ff` -- everything else is expressed in units of it.)
* The step function Gamma(r) is handled exactly: the integration is split into
  segments with a *smooth* right-hand side, and the crossings of r = R_c are
  located with ODE-solver events (instead of an if-statement inside the RHS,
  which makes the RHS discontinuous inside a step of the adaptive integrator).
* The fall-back of the element onto the stellar surface (r = R_*) is detected
  with an event and stops the integration (the 1/r^2 force is singular at 0).
* Gamma and R_c are *computed* from Table 1, never typed in by hand.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # write files only; works on machines without a display
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import constants as const
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# =============================================================================
# 1. Physical constants and parameters (SI units)
# =============================================================================
G = const.G                        # gravitational constant [m^3 kg^-1 s^-2]
C = const.c                        # speed of light [m/s]
SIGMA_SB = const.Stefan_Boltzmann  # Stefan-Boltzmann constant [W m^-2 K^-4]
AU = const.au                      # astronomical unit [m]
YEAR = const.Julian_year           # Julian year, 365.25 d [s]
M_SUN = 1.98841e30                 # solar mass [kg]  (IAU nominal GM_sun / G)
L_SUN = 3.828e26                   # solar luminosity [W] (IAU 2015 nominal)
R_SUN = 6.957e8                    # solar radius [m] (IAU 2015 nominal)

# Numerical settings
RTOL, ATOL = 1e-10, 1e-12          # DOP853 tolerances
MAX_SEGMENTS = 1000                # safety limit for regime switches

# Initial conditions shared by ALL runs (Fig. 1 caption): r0 = R_*, u0 = 0.8 u_esc
R0 = 1.0                           # in units of R_*
U0 = 0.8                           # in units of u_esc(R_*)

# Integration length: 25 free-fall times for the trajectory plots (task 2/3);
# longer for the velocity profile so that r reaches > 45 AU (task 4/5).
T_END_TRAJ = 25.0
T_END_WIND = 150.0

# Planetary orbits: semi-major axes [AU] (Ceres = largest body of the asteroid belt)
PLANETS = [
    ("Mercury", 0.387), ("Venus", 0.723), ("Earth", 1.000), ("Mars", 1.524),
    ("Ceres", 2.767), ("Jupiter", 5.203), ("Saturn", 9.537),
    ("Uranus", 19.19), ("Neptune", 30.07), ("Pluto", 39.48),
]


@dataclass(frozen=True)
class Star:
    """Stellar parameters and derived scales."""
    M: float   # mass [kg]
    L: float   # luminosity [W]
    T: float   # effective temperature [K]

    @property
    def R(self) -> float:
        """Stellar radius from L = 4 pi R^2 sigma T^4 [m]."""
        return float(np.sqrt(self.L / (4.0 * np.pi * SIGMA_SB * self.T ** 4)))

    @property
    def u_esc(self) -> float:
        """Escape velocity at r = R_* [m/s]."""
        return float(np.sqrt(2.0 * G * self.M / self.R))

    @property
    def t_ff(self) -> float:
        """Time scale R_* / u_esc [s] (see module docstring)."""
        return self.R / self.u_esc

    @property
    def g0(self) -> float:
        """Gravitational acceleration at R_* [m/s^2]."""
        return G * self.M / self.R ** 2


@dataclass(frozen=True)
class Grain:
    """Amorphous-carbon grain and gas properties (Table 1 of the assignment)."""
    T_c: float = 1500.0       # condensation temperature [K]
    p: float = 1.0            # power-law index of kappa_abs ~ lambda^-p
    Q_over_a: float = 2.0e6   # Q_rp / a_gr near the flux maximum [1/m]
    A_mon: float = 12.0       # atomic weight of the monomer (C)
    rho_grain: float = 1.85e3  # density of the grain material [kg/m^3]
    eps_c: float = 3.3e-4     # abundance of excess carbon not bound in CO
    eps_He: float = 0.1       # He abundance (by number)
    f_c: float = 0.2          # degree of condensation


# The star used in all runs of the assignment
STAR = Star(M=1.5 * M_SUN, L=7000.0 * L_SUN, T=2600.0)


# =============================================================================
# 2. Dust model: condensation distance, opacity, Gamma
# =============================================================================
def condensation_radius(T_c: float, T_star: float, p: float) -> float:
    """R_c / R_*  (Eq. 2 of the assignment):  (1/2) (T_c/T_*)^(-(4+p)/2)."""
    return 0.5 * (T_c / T_star) ** (-(4.0 + p) / 2.0)


def dust_opacity(g: Grain) -> float:
    """Dust opacity kappa_rp [m^2/kg] (Eq. 3 of the assignment).

    kappa = 3/4 * A_mon/rho_grain * (Q/a) * eps_c/(1 + 4 eps_He) * f_c
    (the proton mass m_p cancels between the monomer volume and n_H).
    """
    return (0.75 * g.A_mon / g.rho_grain * g.Q_over_a
            * g.eps_c / (1.0 + 4.0 * g.eps_He) * g.f_c)


def gamma_factor(kappa_H: float, star: Star) -> float:
    """Gamma = kappa_H L_* / (4 pi c G M_*)  (Eq. 1 of the assignment)."""
    return kappa_H * star.L / (4.0 * np.pi * C * G * star.M)


# =============================================================================
# 3. Numerical integration (dimensionless units)
# =============================================================================
@dataclass
class Trajectory:
    """Result of `integrate_trajectory` (all quantities dimensionless)."""
    Gamma: float
    Rc: float
    t: np.ndarray          # time [t_ff]
    r: np.ndarray          # radial distance [R_*]
    u: np.ndarray          # radial velocity [u_esc]
    segments: list         # (t_start, t_end, dense-output callable) per segment
    fell_back: bool = False
    t_fall: float | None = None   # time of return to r = R_*
    t_apex: float | None = None   # time of the first velocity reversal
    r_apex: float | None = None   # distance at that moment


def integrate_trajectory(Gamma: float, Rc: float, u0: float = U0,
                         r0: float = R0, t_end: float = T_END_TRAJ,
                         n_samples: int = 3000) -> Trajectory:
    """Integrate dr/dt = u, du/dt = -(1 - Gamma_eff(r)) / (2 r^2).

    Gamma_eff = 0 for r < Rc and = Gamma for r >= Rc (step function).
    The integration is split at every crossing of r = Rc (located by an event)
    so that the right-hand side is smooth within each solver call. The run
    stops at t_end or when the element falls back onto the stellar surface.
    """
    t_grid = np.linspace(0.0, t_end, n_samples)
    t_cur, y_cur = 0.0, np.array([r0, u0], dtype=float)
    in_dust = r0 >= Rc
    T, R, U = [0.0], [r0], [u0]
    traj = Trajectory(Gamma=Gamma, Rc=Rc, t=np.empty(0), r=np.empty(0),
                      u=np.empty(0), segments=[])

    for _ in range(MAX_SEGMENTS):
        g_eff = Gamma if in_dust else 0.0

        def rhs(t, y, g=g_eff):
            return [y[1], -(1.0 - g) / (2.0 * y[0] ** 2)]

        def hit_surface(t, y):
            return y[0] - 1.0
        hit_surface.terminal, hit_surface.direction = True, -1

        def cross_Rc(t, y):
            return y[0] - Rc
        cross_Rc.terminal = True
        cross_Rc.direction = -1 if in_dust else +1

        def apex(t, y):
            return y[1]
        apex.terminal, apex.direction = False, -1

        sol = solve_ivp(rhs, (t_cur, t_end), y_cur, method="DOP853",
                        rtol=RTOL, atol=ATOL, dense_output=True,
                        events=[hit_surface, cross_Rc, apex])
        t_seg_end = sol.t[-1]
        traj.segments.append((t_cur, t_seg_end, sol.sol))

        # Uniform samples inside the segment, plus its exact end point
        tg = t_grid[(t_grid > t_cur) & (t_grid < t_seg_end)]
        if tg.size:
            yy = sol.sol(tg)
            T.extend(tg); R.extend(yy[0]); U.extend(yy[1])
        T.append(t_seg_end); R.append(sol.y[0, -1]); U.append(sol.y[1, -1])

        if traj.t_apex is None and sol.t_events[2].size:
            traj.t_apex = float(sol.t_events[2][0])
            traj.r_apex = float(sol.sol(traj.t_apex)[0])

        if sol.status == 0:                      # reached t_end
            break
        if sol.t_events[0].size:                 # fell back onto the star
            traj.fell_back, traj.t_fall = True, float(t_seg_end)
            break
        in_dust = not in_dust                    # crossed R_c: switch regime
        t_cur, y_cur = t_seg_end, sol.y[:, -1].copy()
    else:
        raise RuntimeError("Too many crossings of R_c; check parameters.")

    traj.t, traj.r, traj.u = np.array(T), np.array(R), np.array(U)
    return traj


def state_at_radius(traj: Trajectory, r_target: float, tol: float = 1e-9):
    """First time the element is at r = r_target: returns (t, r, u).

    Uses the dense output of the solver and a root finder, so no interpolation
    error from a coarse output grid enters the comparison with theory.
    """
    for t0, t1, f in traj.segments:
        r_a, r_b = f(t0)[0], f(t1)[0]
        if abs(r_a - r_target) < tol:
            s = f(t0); return t0, s[0], s[1]
        if abs(r_b - r_target) < tol:
            s = f(t1); return t1, s[0], s[1]
        if (r_a - r_target) * (r_b - r_target) < 0.0:
            t_x = brentq(lambda t: f(t)[0] - r_target, t0, t1,
                         xtol=1e-14, rtol=1e-14)
            s = f(t_x); return t_x, s[0], s[1]
    raise ValueError(f"r = {r_target} is not reached by this trajectory")


# =============================================================================
# 4. Analytical results
# =============================================================================
def u_analytic(r, Gamma: float, Rc: float, u0: float = U0):
    """Outward velocity [u_esc] from Eq. (4), piecewise in r.

    r < Rc:  u^2 = u0^2 + 1/r - 1                     (Gamma = 0, free fall)
    r >= Rc: u^2 = u(Rc)^2 + (1 - Gamma)(1/r - 1/Rc)  (Gamma = const)
    Returns NaN where u^2 < 0 (i.e. beyond the turning point).
    """
    r = np.asarray(r, dtype=float)
    u2_Rc = u0 ** 2 + 1.0 / Rc - 1.0
    u2 = np.where(r < Rc, u0 ** 2 + 1.0 / r - 1.0,
                  u2_Rc + (1.0 - Gamma) * (1.0 / r - 1.0 / Rc))
    return np.sqrt(np.where(u2 >= 0.0, u2, np.nan))


def terminal_velocity(Gamma: float, Rc: float, u0: float = U0) -> float:
    """lim_{r->inf} u [u_esc] = sqrt(u(Rc)^2 + (Gamma - 1)/Rc).

    NaN if the element never reaches R_c or does not escape.
    """
    u2_Rc = u0 ** 2 + 1.0 / Rc - 1.0
    if u2_Rc < 0.0:
        return float("nan")
    u2_inf = u2_Rc + (Gamma - 1.0) / Rc
    return float(np.sqrt(u2_inf)) if u2_inf > 0.0 else float("nan")


def escape_radius(Gamma: float, Rc: float, u0: float = U0) -> float:
    """Distance [R_*] beyond which u(r) > u_esc(r) = u_esc(R_*) / sqrt(r).

    For r >= Rc: u^2 = u_inf^2 - (Gamma - 1)/r  and  u_esc(r)^2 = 1/r, so the
    two are equal at r_esc = Gamma / u_inf^2. (For r < Rc, u^2 < u_esc^2 always
    because u0 < u_esc.)
    """
    u_inf = terminal_velocity(Gamma, Rc, u0)
    return Gamma / u_inf ** 2 if np.isfinite(u_inf) else float("nan")


def free_fall_apex(u0: float = U0):
    """Gamma = 0: analytical apex distance, apex time and fall-back time.

    r_max = 1/(1 - u0^2)  (Eq. 3 of the lecture notes). The motion is a radial
    Kepler orbit with semi-major axis a = r_max/2, r = a (1 - cos eta),
    t = sqrt(a^3/GM) (eta - sin eta) with GM = 1/2 in these units.
    Times are measured from r = R_*.
    """
    r_max = 1.0 / (1.0 - u0 ** 2)
    a = 0.5 * r_max
    eta0 = np.arccos(1.0 - R0 / a)
    pref = np.sqrt(a ** 3 / 0.5)
    t_apex = pref * (np.pi - (eta0 - np.sin(eta0)))
    return r_max, t_apex, 2.0 * t_apex


def km_s(u_dimless, star: Star = STAR):
    """Convert a dimensionless velocity to km/s."""
    return np.asarray(u_dimless) * star.u_esc / 1.0e3


def to_years(t_dimless, star: Star = STAR):
    """Convert dimensionless time to years."""
    return np.asarray(t_dimless) * star.t_ff / YEAR


# =============================================================================
# 5. Reporting helper
# =============================================================================
class Report:
    """Print to stdout and keep the lines for the summary file."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# =============================================================================
# 6. Code tests
# =============================================================================
def run_code_tests(report: Report, Gamma_phys: float, Rc_phys: float) -> bool:
    """Tests of the formulae and of the numerical integration."""
    results: list[bool] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append(bool(ok))
        report(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    report("CODE TESTS")
    report("-" * 78)

    # (a) Formulae vs. statements in the lecture notes
    Rc_3000 = condensation_radius(1500.0, 3000.0, 1.0)
    check("R_c for T_*=3000 K, T_c=1500 K, p=1", 2.0 <= Rc_3000 <= 3.0,
          f"{Rc_3000:.3f} R_* (notes: 'R_c ~ 2-3 R_*')")
    g_notes = gamma_factor(dust_opacity(Grain(f_c=1.0)),
                           Star(M=M_SUN, L=5000.0 * L_SUN, T=3000.0))
    check("Gamma for notes example (f_c=1, 5000 L_sun, 1 M_sun)",
          8.0 <= g_notes <= 11.0, f"{g_notes:.2f} (notes: 'Gamma ~ 10')")

    # (b) Gamma = 0: compare with the analytical free-fall solution
    tr0 = integrate_trajectory(0.0, 2.5)
    r_max, t_apex, t_fall = free_fall_apex()
    err = abs(tr0.r_apex - r_max) / r_max
    check("Gamma=0: apex distance", err < 1e-8,
          f"numerical {tr0.r_apex:.10f}, analytical {r_max:.10f}, rel. err {err:.1e}")
    err = abs(tr0.t_apex - t_apex) / t_apex
    check("Gamma=0: apex time", err < 1e-8,
          f"numerical {tr0.t_apex:.8f} t_ff, analytical {t_apex:.8f} t_ff, rel. err {err:.1e}")
    err = abs(tr0.t_fall - t_fall) / t_fall if tr0.fell_back else np.inf
    check("Gamma=0: fall-back time", err < 1e-8,
          f"numerical {tr0.t_fall:.8f} t_ff, analytical {t_fall:.8f} t_ff, rel. err {err:.1e}")

    # (c) R_c beyond the apex: dust is never reached, trajectory = free fall
    tr3 = integrate_trajectory(5.0, 3.0)
    check("R_c=3.0 > r_max: element never reaches the dust zone",
          tr3.fell_back and tr3.r_apex < 3.0 and
          abs(tr3.t_fall - tr0.t_fall) < 1e-8,
          f"r_max = {tr3.r_apex:.4f} < 3.0, falls back at t = {tr3.t_fall:.5f} t_ff")

    # (d) Gamma = 1: gravity and radiation pressure cancel -> constant velocity
    tr1 = integrate_trajectory(1.0, 2.5)
    _, _, u_at_Rc = state_at_radius(tr1, 2.5)
    dev = np.max(np.abs(tr1.u[tr1.r >= 2.5] - u_at_Rc))
    check("Gamma=1: constant velocity beyond R_c", dev < 1e-8,
          f"u(R_c) = {u_at_Rc:.8f} u_esc, max |u - u(R_c)| = {dev:.1e}")

    # (e) Gamma = 5, R_c = 2.5: pointwise comparison with Eq. (4)
    tr5 = integrate_trajectory(5.0, 2.5)
    mask = tr5.r < 60.0
    rel = np.abs(tr5.u[mask] - u_analytic(tr5.r[mask], 5.0, 2.5)) / tr5.u[mask]
    check("Gamma=5, R_c=2.5: u(r) vs Eq. (4) on all output points",
          rel.max() < 1e-8, f"max rel. err {rel.max():.1e}")

    # (f) Terminal velocity formula = large-r limit of Eq. (4)
    u_inf = terminal_velocity(Gamma_phys, Rc_phys)
    u_far = float(u_analytic(1e12, Gamma_phys, Rc_phys))
    check("Terminal velocity formula vs Eq. (4) at r = 1e12 R_*",
          abs(u_far - u_inf) / u_inf < 1e-9,
          f"u_inf = {u_inf:.8f} u_esc, u(1e12 R_*) = {u_far:.8f} u_esc")

    ok = all(results)
    report("-" * 78)
    report(f"{sum(results)}/{len(results)} tests passed.")
    report()
    return ok


# =============================================================================
# 7. Plotting helpers
# =============================================================================
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.linestyle": ":",
                     "grid.alpha": 0.6, "legend.fontsize": 9.5,
                     "figure.dpi": 100})

STAR_TITLE = (r"$M_*=1.5\,M_\odot$, $L_*=7000\,L_\odot$, $T_*=2600$ K")


def case_label(Gamma: float, Rc: float, nd: int = 1) -> str:
    return rf"$\Gamma = {Gamma:.{nd}f}$, $R_c = {Rc:.{nd}f}\,R_*$"


def save_fig(fig, outdir: Path, name: str, dpi: int) -> Path:
    path = outdir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_reproduce_fig1(outdir: Path, dpi: int) -> Path:
    """Task 2: reproduce Fig. 1 (integrated to 25 t_ff, shown for 0-2.7 yr)."""
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 9.5))
    panels = [
        ([(5.0, 2.5), (1.0, 2.5), (0.0, 2.5)], "Varying $\\Gamma$ (fixed $R_c$)"),
        ([(5.0, 2.0), (5.0, 2.5), (5.0, 3.0)], "Varying $R_c$ (fixed $\\Gamma$)"),
    ]
    for ax, (cases, subtitle) in zip(axes, panels):
        for Gamma, Rc in cases:
            tr = integrate_trajectory(Gamma, Rc, t_end=T_END_TRAJ)
            ax.plot(to_years(tr.t), tr.r, label=case_label(Gamma, Rc))
        ax.set_xlim(0.0, 2.7)
        ax.set_ylim(1.0, 7.0)
        ax.set_xlabel("Time [yr]")
        ax.set_ylabel(r"Radial distance [$R_*$]")
        ax.set_title(STAR_TITLE + "\n" + subtitle, fontsize=10.5)
        ax.legend(loc="upper left")
    fig.tight_layout()
    return save_fig(fig, outdir, "fig1_test_cases.png", dpi)


def fig_physical_vs_tests(outdir: Path, dpi: int, Gamma_p: float, Rc_p: float,
                          cases_tests) -> Path:
    """Task 3: trajectory for grain-based Gamma, R_c vs. the test cases."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    tr_p = integrate_trajectory(Gamma_p, Rc_p, t_end=T_END_TRAJ)
    for ax in (ax1, ax2):
        for i, (Gamma, Rc) in enumerate(cases_tests):
            tr = integrate_trajectory(Gamma, Rc, t_end=T_END_TRAJ)
            ax.plot(to_years(tr.t), tr.r, ls="--", lw=1.4, color=f"C{i}",
                    label="Test: " + case_label(Gamma, Rc))
        ax.plot(to_years(tr_p.t), tr_p.r, color="k", lw=2.4,
                label="Grain-based: " + case_label(Gamma_p, Rc_p, nd=2))
        ax.axhline(Rc_p, color="0.5", ls=":", lw=1.0)
        ax.set_xlabel("Time [yr]")
        ax.set_ylabel(r"Radial distance [$R_*$]")
    ax1.set_xlim(0.0, 2.7); ax1.set_ylim(1.0, 7.0)
    ax1.set_title("Same window as Fig. 1")
    ax1.legend(loc="upper left")
    ax2.set_xlim(0.0, float(to_years(T_END_TRAJ)))
    ax2.set_yscale("log"); ax2.set_ylim(1.0, 100.0)
    ax2.set_title(r"Full integration (25 $t_{ff}$), log scale")
    ax2.text(0.02, 0.97, rf"dotted: $R_c = {Rc_p:.2f}\,R_*$ (grain-based)",
             transform=ax2.transAxes, va="top", fontsize=9, color="0.3")
    fig.suptitle(STAR_TITLE, fontsize=11)
    fig.tight_layout()
    return save_fig(fig, outdir, "fig2_grain_model_vs_tests.png", dpi)


def fig_velocity_vs_radius(outdir: Path, dpi: int, traj: Trajectory,
                           table_rows: list, Gamma: float, Rc: float,
                           u_inf: float) -> Path:
    """Task 4: u(r) in km/s vs r/R_*, numerical vs analytical, with residuals."""
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8.5, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08})
    sel = (traj.r >= 1.0) & (traj.r <= 25.0)
    r_grid = np.linspace(1.0, 25.0, 600)
    ax.plot(traj.r[sel], km_s(traj.u[sel]), "k-", lw=2.2,
            label="Numerical solution (ODE, DOP853)")
    ax.plot(r_grid, km_s(1.0 / np.sqrt(r_grid)), color="C3", ls="--", lw=1.3,
            alpha=0.8, label=r"Local escape velocity $u_{\rm esc}(r)$")
    ax.scatter([row["r"] for row in table_rows],
               [row["u_ana"] for row in table_rows], c="C3", s=46, zorder=5,
               edgecolor="w", label="Analytical, Eq. (4), checkpoints")
    ax.axhline(km_s(u_inf), color="C0", ls="-.", lw=1.3,
               label=rf"Terminal velocity $u_\infty = {km_s(u_inf):.1f}$ km/s")
    ax.axvline(Rc, color="0.45", ls="--", lw=1.1,
               label=rf"Condensation distance $R_c = {Rc:.2f}\,R_*$")
    ax.set_ylabel("Velocity [km/s]")
    ax.set_ylim(0.0, 1.45 * float(km_s(1.0)))
    ax.set_title("Dust-driven wind: velocity vs. distance\n" + STAR_TITLE
                 + rf", $\Gamma={Gamma:.2f}$", fontsize=10.5)
    ax.legend(loc="upper right")

    rel = np.abs(traj.u[sel] - u_analytic(traj.r[sel], Gamma, Rc)) / traj.u[sel]
    axr.semilogy(traj.r[sel], np.maximum(rel, 1e-17), color="k", lw=1.2)
    axr.axvline(Rc, color="0.45", ls="--", lw=1.1)
    axr.set_xlim(1.0, 25.0)
    axr.set_xlabel(r"Distance from the stellar centre [$R_*$]")
    axr.set_ylabel("|u$_{num}$ - u$_{ana}$| / u$_{ana}$")
    axr.set_ylim(1e-17, 1e-5)
    return save_fig(fig, outdir, "fig3_velocity_vs_radius.png", dpi)


def fig_velocity_AU(outdir: Path, dpi: int, star: Star, traj: Trajectory,
                    Gamma: float, Rc: float, u_inf: float, r_esc: float,
                    pluto_u_kms: float, log_x: bool) -> Path:
    """Task 5: u(r) in km/s vs distance in AU with planetary orbits."""
    R_AU = star.R / AU
    Rc_AU = Rc * R_AU
    r_AU = traj.r * R_AU
    x_min, x_max = (0.3, 60.0) if log_x else (0.0, 45.0)
    sel = (r_AU >= R_AU) & (r_AU <= x_max)
    r_esc_AU = r_esc * R_AU

    fig, ax = plt.subplots(figsize=(11, 6.2))
    # Regions
    ax.axvspan(x_min, R_AU, color="0.85", alpha=0.7, lw=0)
    ax.axvspan(R_AU, Rc_AU, color="#f6c177", alpha=0.35, lw=0)
    ax.axvspan(Rc_AU, x_max, color="#9ccfd8", alpha=0.18, lw=0)

    ax.plot(r_AU[sel], km_s(traj.u[sel]), "k-", lw=2.4, label=r"Wind velocity $u(r)$")
    r_plot = np.geomspace(R_AU, x_max, 800) if log_x else np.linspace(R_AU, x_max, 800)
    ax.plot(r_plot, km_s(1.0) * np.sqrt(R_AU / r_plot), color="C3", ls="--",
            lw=1.6, label=r"Local escape velocity $u_{\rm esc}(r)$")
    ax.axhline(km_s(u_inf), color="C0", ls="-.", lw=1.3,
               label=rf"Terminal velocity $u_\infty = {km_s(u_inf):.1f}$ km/s")
    ax.plot([r_esc_AU], [float(km_s(1.0 / np.sqrt(r_esc)))], "o", color="C3",
            ms=8, mec="k", zorder=6,
            label=rf"$u = u_{{\rm esc}}$ at $r = {r_esc_AU:.1f}$ AU ({r_esc:.2f} $R_*$)")
    pl_au = dict(PLANETS)["Pluto"]
    ax.plot([pl_au], [pluto_u_kms], "s", color="C2", ms=8, mec="k", zorder=6,
            label=rf"Pluto's orbit ({pl_au:.1f} AU): {pluto_u_kms:.2f} km/s "
                  rf"({100 * pluto_u_kms / km_s(u_inf):.1f}% of $u_\infty$)")

    # Planetary orbits
    trans = ax.get_xaxis_transform()
    group_done = False
    for name, a in PLANETS:
        ax.axvline(a, color="0.35", lw=0.7, ls=":", zorder=1)
        if not log_x and a < 1.6:
            if not group_done:
                ax.text(0.95, 0.985, "Mercury, Venus, Earth, Mars", rotation=90,
                        ha="center", va="top", fontsize=7.5, transform=trans)
                group_done = True
            continue
        ax.text(a, 0.985, name, rotation=90, ha="right", va="top",
                fontsize=8, transform=trans)

    # Region labels
    if log_x:
        ax.text(0.62, 0.04, "Inside star", ha="center", transform=trans, fontsize=9)
        ax.text(np.sqrt(R_AU * Rc_AU), 0.04, "Dust-free\natmosphere", ha="center",
                transform=trans, fontsize=9)
    else:
        ax.text(R_AU / 2, 0.04, "Inside star", rotation=90, ha="center",
                transform=trans, fontsize=9)
        ax.text((R_AU + Rc_AU) / 2, 0.04, "Dust-free atmosphere", rotation=90,
                ha="center", transform=trans, fontsize=9)
    ax.text(14.0 if not log_x else 15.0, 0.04, "Dusty wind", ha="center",
            transform=trans, fontsize=9)

    ax.set_xlim(x_min, x_max)
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks([0.3, 1, 2, 5, 10, 20, 50])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0.0, 1.12 * float(km_s(1.0)))
    ax.set_xlabel("Distance from the stellar centre [AU]")
    ax.set_ylabel("Velocity [km/s]")
    ax.set_title("Dust-driven wind in astronomical units\n" + STAR_TITLE
                 + rf", $\Gamma={Gamma:.2f}$, $R_c={Rc:.2f}\,R_*={Rc_AU:.2f}$ AU "
                 + rf"($R_*={R_AU:.2f}$ AU)", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              framealpha=0.95)
    name = "fig5_velocity_AU_log.png" if log_x else "fig4_velocity_AU.png"
    return save_fig(fig, outdir, name, dpi)


# =============================================================================
# 8. Main program
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--outdir", default="wind_output", help="output directory")
    ap.add_argument("--dpi", type=int, default=200, help="figure resolution")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = Report()
    star, grain = STAR, Grain()

    # ---- Stellar scales and grain-based parameters ----------------------------
    Rc_phys = condensation_radius(grain.T_c, star.T, grain.p)
    kappa_H = dust_opacity(grain)
    Gamma_phys = gamma_factor(kappa_H, star)
    u_inf = terminal_velocity(Gamma_phys, Rc_phys)
    r_esc = escape_radius(Gamma_phys, Rc_phys)
    R_AU = star.R / AU

    report("=" * 78)
    report("A TOY MODEL FOR DUST-DRIVEN WINDS -- RESULTS SUMMARY")
    report("=" * 78)
    report("STELLAR PARAMETERS AND SCALES")
    report(f"  M_* = 1.5 M_sun = {star.M:.4e} kg,  L_* = 7000 L_sun = {star.L:.4e} W,  T_* = {star.T:.0f} K")
    report(f"  R_*        = {star.R:.4e} m = {star.R / R_SUN:.1f} R_sun = {R_AU:.4f} AU")
    report(f"  u_esc(R_*) = {star.u_esc / 1e3:.3f} km/s")
    report(f"  t_ff       = R_*/u_esc = {star.t_ff:.4e} s = {star.t_ff / YEAR:.4f} yr")
    report(f"  25 t_ff    = {25 * star.t_ff / YEAR:.3f} yr")
    report(f"  g_0        = {star.g0:.4e} m/s^2")
    report(f"  Initial conditions: r_0 = R_*, u_0 = {U0} u_esc = {U0 * star.u_esc / 1e3:.2f} km/s")
    report()
    report("GRAIN-BASED PARAMETERS (Table 1, amorphous carbon)")
    report(f"  R_c/R_* = 0.5 (T_c/T_*)^(-(4+p)/2) = {Rc_phys:.4f}   ({Rc_phys * R_AU:.3f} AU)")
    report(f"  kappa_H = {kappa_H:.4f} m^2/kg")
    report(f"  Gamma   = kappa_H L_*/(4 pi c G M_*) = {Gamma_phys:.4f}")
    r_max_ff, _, _ = free_fall_apex()
    report(f"  Free-fall apex (Gamma=0): r_max = {r_max_ff:.4f} R_*  "
           f"{'>' if r_max_ff > Rc_phys else '<='} R_c  -> the element "
           f"{'reaches' if r_max_ff > Rc_phys else 'does not reach'} the dust-forming zone")
    report()

    # ---- Tests ------------------------------------------------------------------
    tests_ok = run_code_tests(report, Gamma_phys, Rc_phys)

    # ---- Tasks 1-2: reproduction of Fig. 1 ---------------------------------------
    p1 = fig_reproduce_fig1(outdir, args.dpi)
    report(f"Task 2: test cases written to {p1}")
    report("  Fig. 1 of the assignment shows only the first ~2.7 yr (~11 t_ff); the "
           "integration itself runs to 25 t_ff.")
    report()

    # ---- Task 3: grain-based trajectory vs. test cases -------------------------
    cases_tests = [(0.0, 2.5), (1.0, 2.5), (5.0, 2.5), (5.0, 2.0)]
    p2 = fig_physical_vs_tests(outdir, args.dpi, Gamma_phys, Rc_phys, cases_tests)
    report(f"Task 3: comparison figure written to {p2}")
    report("  Time [yr] needed to reach r = 7 R_* (within 25 t_ff):")
    for Gamma, Rc, tag in ([(Gamma_phys, Rc_phys, "grain-based")]
                           + [(g, rc, "test") for g, rc in cases_tests]):
        tr = integrate_trajectory(Gamma, Rc, t_end=T_END_TRAJ)
        try:
            t7 = float(to_years(state_at_radius(tr, 7.0)[0]))
            txt = f"{t7:.2f} yr"
        except ValueError:
            txt = ("falls back onto the star at "
                   f"{float(to_years(tr.t_fall)):.2f} yr" if tr.fell_back
                   else f"not reached (r = {tr.r[-1]:.2f} R_* at 25 t_ff)")
        report(f"    {tag:11s} Gamma = {Gamma:5.2f}, R_c = {Rc:5.2f}:  {txt}")
    report()

    # ---- Task 4: velocity structure, numerical vs. analytical -----------------
    wind = integrate_trajectory(Gamma_phys, Rc_phys, t_end=T_END_WIND, n_samples=9000)
    check_radii = [1.25, 1.5, 1.9, Rc_phys, 2.1, 3.0, 5.0, 10.0, 20.0]
    rows = []
    for r in check_radii:
        _, r_num, u_num = state_at_radius(wind, r)
        u_a = float(u_analytic(r, Gamma_phys, Rc_phys))
        region = "dust-free" if r < Rc_phys - 1e-9 else ("R_c" if abs(r - Rc_phys) < 1e-9 else "dusty")
        rows.append({"r": r, "region": region, "u_num": float(km_s(u_num)),
                     "u_ana": float(km_s(u_a)),
                     "rel_err": abs(u_num - u_a) / u_a})
    report("TASK 4: NUMERICAL vs. ANALYTICAL VELOCITY (Eq. 4), grain-based model")
    report("-" * 78)
    report(f"  {'r [R_*]':>9} {'region':>10} {'u_num [km/s]':>14} {'u_ana [km/s]':>14} {'rel. error':>12}")
    for row in rows:
        report(f"  {row['r']:9.4f} {row['region']:>10} {row['u_num']:14.6f} "
               f"{row['u_ana']:14.6f} {row['rel_err']:12.2e}")
    report("-" * 78)
    u_Rc = float(km_s(u_analytic(Rc_phys, Gamma_phys, Rc_phys)))
    report(f"  u(R_c) = {u_Rc:.3f} km/s  (decelerated from u_0 = {U0 * star.u_esc / 1e3:.2f} km/s by gravity)")
    report(f"  Terminal velocity: u_inf^2 = u(R_c)^2 + (Gamma - 1) u_esc^2 / (R_c/R_*)")
    report(f"                     u_inf = {u_inf:.5f} u_esc = {float(km_s(u_inf)):.3f} km/s")
    report(f"  Numerical u at r = {wind.r[-1]:.1f} R_* (t = {wind.t[-1]:.0f} t_ff): "
           f"{float(km_s(wind.u[-1])):.3f} km/s")
    report()
    p3 = fig_velocity_vs_radius(outdir, args.dpi, wind, rows, Gamma_phys, Rc_phys, u_inf)
    report(f"Task 4: velocity figure written to {p3}")
    report()

    with open(outdir / "velocity_check_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["r_over_Rstar", "r_AU", "region", "u_numerical_kms",
                    "u_analytical_kms", "relative_error"])
        for row in rows:
            w.writerow([f"{row['r']:.6f}", f"{row['r'] * R_AU:.6f}", row["region"],
                        f"{row['u_num']:.8f}", f"{row['u_ana']:.8f}", f"{row['rel_err']:.3e}"])

    # ---- Task 5: astrophysical interpretation ---------------------------------
    report("TASK 5: ASTROPHYSICAL INTERPRETATION")
    report("-" * 78)
    Rc_AU = Rc_phys * R_AU
    report(f"  Star: 0 - {R_AU:.2f} AU | dust-free atmosphere: {R_AU:.2f} - {Rc_AU:.2f} AU | "
           f"dusty wind: > {Rc_AU:.2f} AU")
    report(f"  Wind exceeds the local escape velocity for r > {r_esc:.3f} R_* = {r_esc * R_AU:.2f} AU "
           f"(r_esc = Gamma/u_inf^2, u_inf in u_esc)")
    # Numerical cross-check of the escape radius
    d = wind.u - 1.0 / np.sqrt(wind.r)
    k = int(np.argmax(d > 0))
    r_esc_num = wind.r[k - 1] + (wind.r[k] - wind.r[k - 1]) * (-d[k - 1]) / (d[k] - d[k - 1])
    report(f"  (numerical cross-check from the integrated solution: {r_esc_num:.3f} R_*)")
    report()
    report(f"  {'Body':<9} {'a [AU]':>8} {'a [R_*]':>9}  {'region':<20} {'u_wind [km/s]':>14} "
           f"{'u_esc [km/s]':>13}  {'u > u_esc?':>10}")
    for name, a in PLANETS:
        r = a / R_AU
        if r < 1.0:
            region, uw, flag = "inside the star", "-", "-"
        else:
            region = "dust-free atmosphere" if r < Rc_phys else "dusty wind"
            u_w = float(km_s(u_analytic(r, Gamma_phys, Rc_phys)))
            uw = f"{u_w:.2f}"
            flag = "yes" if u_w > float(km_s(1.0 / np.sqrt(r))) else "no"
        ue = f"{float(km_s(1.0 / np.sqrt(r))):.2f}" if r >= 1.0 else "-"
        report(f"  {name:<9} {a:8.3f} {r:9.3f}  {region:<20} {uw:>14} {ue:>13}  {flag:>10}")
    report()
    pl_au = dict(PLANETS)["Pluto"]
    _, _, u_pl = state_at_radius(wind, pl_au / R_AU)
    u_pl_kms = float(km_s(u_pl))
    report(f"  Pluto ({pl_au} AU = {pl_au / R_AU:.2f} R_*): numerical u = {u_pl_kms:.2f} km/s, "
           f"analytical u = {float(km_s(u_analytic(pl_au / R_AU, Gamma_phys, Rc_phys))):.2f} km/s "
           f"= {100 * u_pl_kms / float(km_s(u_inf)):.1f}% of u_inf = {float(km_s(u_inf)):.2f} km/s")
    report()
    p4 = fig_velocity_AU(outdir, args.dpi, star, wind, Gamma_phys, Rc_phys, u_inf,
                         r_esc, u_pl_kms, log_x=False)
    p5 = fig_velocity_AU(outdir, args.dpi, star, wind, Gamma_phys, Rc_phys, u_inf,
                         r_esc, u_pl_kms, log_x=True)
    report(f"Task 5: figures written to {p4} and {p5}")
    report()

    report_path = outdir / "results_summary.txt"
    report("All tests passed." if tests_ok else "WARNING: some tests FAILED (see above).")
    report.save(report_path)
    print(f"Summary saved to {report_path}")
    return 0 if tests_ok else 1


if __name__ == "__main__":
    sys.exit(main())
