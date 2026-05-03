"""Compare delta-method vs Monte-Carlo propagation of the velocity covariance
through the speed/clock/deflection-angle transforms in
``derive_velocity_angles``.

For each test case (a velocity vector ``u`` in the DSRF frame and a 3x3
covariance) the script prints:
  - delta-method σ (current closed-form linearization)
  - MC σ at N=1000 (the proposed production path)
  - MC σ at N=200_000 (treated as ground truth)
  - relative bias of the delta vs ground truth
  - relative noise of MC@1000 vs ground truth (sampling error)

The point of the comparison is to decide whether MC is also worth using for
the speed sigma, or whether delta is accurate enough there.

Run: ``conda run -n imapL3 python scripts/swapi/compare_angle_propagation.py``
"""

from __future__ import annotations

import numpy as np


def delta_method_sigmas(u: np.ndarray, cov: np.ndarray) -> tuple[float, float, float]:
    speed = float(np.linalg.norm(u))
    speed2 = speed**2
    vxy2 = float(u[0] ** 2 + u[1] ** 2)
    vxy = float(np.sqrt(vxy2))

    g_speed = u / speed
    speed_sigma = float(np.sqrt(g_speed @ cov @ g_speed))

    if vxy2 > 0:
        g_clock = np.array([-u[1] / vxy2, u[0] / vxy2, 0.0])
        clock_sigma = float(np.degrees(np.sqrt(g_clock @ cov @ g_clock)))
        g_defl = np.array(
            [u[0] * u[2] / (speed2 * vxy), u[1] * u[2] / (speed2 * vxy), -vxy / speed2]
        )
        defl_sigma = float(np.degrees(np.sqrt(g_defl @ cov @ g_defl)))
    else:
        clock_sigma = np.nan
        defl_sigma = np.nan
    return speed_sigma, clock_sigma, defl_sigma


def mc_sigmas(
    u: np.ndarray, cov: np.ndarray, n: int, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(u, cov, size=n, check_valid="ignore")
    speeds = np.linalg.norm(samples, axis=1)
    speed_sigma = float(np.std(speeds, ddof=1))

    clock_nom = float(np.degrees(np.arctan2(u[1], u[0])) % 360.0)
    clocks = np.degrees(np.arctan2(samples[:, 1], samples[:, 0])) % 360.0
    clock_resid = ((clocks - clock_nom + 180.0) % 360.0) - 180.0
    clock_sigma = float(np.std(clock_resid, ddof=1))

    defls = np.degrees(np.arccos(np.clip(-samples[:, 2] / speeds, -1.0, 1.0)))
    defl_sigma = float(np.std(defls, ddof=1))
    return speed_sigma, clock_sigma, defl_sigma


def make_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """A spread of (name, u_DSRF [km/s], cov_DSRF [km²/s²]) test cases.

    Spin axis is +z_DSRF; nominal SW points along −z. v_xy is small for
    cold/aligned plasma — that is exactly the regime where the delta-method
    angle gradients (∝ 1/v_xy² and 1/(s²·v_xy)) become large compared to
    higher-order terms.
    """
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    # Cold, well-aligned SW. v_xy ≈ 0, σ ~ 5 km/s isotropic.
    u = np.array([2.0, 1.0, -400.0])
    cov = (5.0**2) * np.eye(3)
    cases.append(("cold-aligned σ_iso=5", u, cov))

    # Cold SW with slight clock-angle offset.
    u = np.array([15.0, 5.0, -400.0])
    cases.append(("cold mild-deflection σ_iso=5", u, cov))

    # Hot plasma — larger covariance, σ comparable to v_xy.
    u = np.array([10.0, 5.0, -400.0])
    cov = (15.0**2) * np.eye(3)
    cases.append(("hot σ_iso=15, v_xy~σ", u, cov))

    # Anisotropic covariance: v_R well-determined, v_T/v_N noisy.
    u = np.array([0.0, 0.0, -400.0])
    cov = np.diag([4.0, 100.0, 100.0])
    cases.append(("aligned σ_x=2 σ_y=σ_z=10", u, cov))

    # Strongly deflected SW (10° off spin axis).
    u = np.array([60.0, 30.0, -395.0])
    cov = (8.0**2) * np.eye(3)
    cases.append(("10°-deflected σ_iso=8", u, cov))

    # Pathological: σ_xy comparable to v_xy.
    u = np.array([5.0, 0.0, -400.0])
    cov = (5.0**2) * np.eye(3)
    cases.append(("pathological σ_xy~v_xy", u, cov))

    # Correlated errors (cov has off-diagonal).
    u = np.array([10.0, 5.0, -400.0])
    cov = np.array(
        [
            [25.0, 10.0, 0.0],
            [10.0, 25.0, 0.0],
            [0.0, 0.0, 25.0],
        ]
    )
    cases.append(("correlated σ_iso=5", u, cov))

    return cases


def main() -> None:
    cases = make_cases()
    print(
        f"{'case':40s}  {'σ_speed [km/s]':>26s}  {'σ_clock [°]':>26s}  {'σ_defl [°]':>26s}"
    )
    print(
        f"{'':40s}  {'delta':>8s} {'MC1k':>8s} {'truth':>8s}"
        f"  {'delta':>8s} {'MC1k':>8s} {'truth':>8s}"
        f"  {'delta':>8s} {'MC1k':>8s} {'truth':>8s}"
    )
    print("-" * 130)

    rows = []
    for name, u, cov in cases:
        d_sp, d_cl, d_de = delta_method_sigmas(u, cov)
        m_sp, m_cl, m_de = mc_sigmas(u, cov, n=1000, seed=0)
        t_sp, t_cl, t_de = mc_sigmas(u, cov, n=200_000, seed=42)
        rows.append((name, (d_sp, m_sp, t_sp), (d_cl, m_cl, t_cl), (d_de, m_de, t_de)))
        print(
            f"{name:40s}  "
            f"{d_sp:8.4f} {m_sp:8.4f} {t_sp:8.4f}  "
            f"{d_cl:8.4f} {m_cl:8.4f} {t_cl:8.4f}  "
            f"{d_de:8.4f} {m_de:8.4f} {t_de:8.4f}"
        )

    print()
    print("Relative bias of delta vs truth (delta/truth − 1, %):")
    print(f"{'case':40s}  {'speed':>10s}  {'clock':>10s}  {'defl':>10s}")
    for name, sp, cl, de in rows:
        bias_sp = 100.0 * (sp[0] / sp[2] - 1.0)
        bias_cl = 100.0 * (cl[0] / cl[2] - 1.0)
        bias_de = 100.0 * (de[0] / de[2] - 1.0)
        print(f"{name:40s}  {bias_sp:9.2f}%  {bias_cl:9.2f}%  {bias_de:9.2f}%")

    print()
    print("MC@1k sampling noise (MC1k/truth − 1, %):")
    print(f"{'case':40s}  {'speed':>10s}  {'clock':>10s}  {'defl':>10s}")
    for name, sp, cl, de in rows:
        noise_sp = 100.0 * (sp[1] / sp[2] - 1.0)
        noise_cl = 100.0 * (cl[1] / cl[2] - 1.0)
        noise_de = 100.0 * (de[1] / de[2] - 1.0)
        print(f"{name:40s}  {noise_sp:9.2f}%  {noise_cl:9.2f}%  {noise_de:9.2f}%")


if __name__ == "__main__":
    main()
