#!/usr/bin/env python3
"""Compare a homogeneous ekpyrotic run with the scaling solution.

Usage: check_homogeneous.py run.log [run2.log ...]

Reads the "[ekpyrotic]" diagnostic lines printed by EkpyroticLevel and
compares chi, K and the lapse with the analytic solution (kappa^2 = 1):

  a ~ |tau|^p,  p = 2 / c^2,  K = 3 p / |tau|  (K > 0 contracting),

with |tau_0| = 3 p / K0 at t = 0. With harmonic slicing
(d_t lapse = -lapse^2 K) the lapse is lapse = a^3 and dt = dtau / a^3, so in
coordinate time, with s = |tau| / |tau_0| and q = 1 - 3 p,

  s(t) = (1 - q t / |tau_0|)^(1/q),
  a = s^p,  chi = a^-2 = s^(-2 p),  K = K0 / s,  lapse = a^3 = s^(3 p),

and the big crunch is at t_c = |tau_0| / q.
"""

import math
import sys


def read_log(path):
    rows = []
    with open(path) as log:
        for line in log:
            if not line.startswith("[ekpyrotic]"):
                continue
            tokens = line.split()[1:]
            values = dict(zip(tokens[0::2], tokens[1::2]))
            rows.append({key: float(value) for key, value in values.items()})
    return rows


def analytic(t, K0, p):
    tau0 = 3.0 * p / K0
    q    = 1.0 - 3.0 * p
    s    = (1.0 - q * t / tau0) ** (1.0 / q)
    return {"chi": s ** (-2.0 * p), "K": K0 / s, "lapse": s ** (3.0 * p)}


def main():
    c = 10.0  # scalar_field.c in params.txt
    p = 2.0 / c**2
    for path in sys.argv[1:]:
        rows = read_log(path)
        K0   = rows[0]["K_mean"]
        tau0 = 3.0 * p / K0
        print(f"{path}: p = {p}, K0 = {K0:.10e}, |tau0| = {tau0:.6f}, "
              f"t_crunch = {tau0 / (1 - 3 * p):.6f}")
        print(f"{'t':>8} {'K/K0':>10} {'rel_err_K':>10} {'rel_err_chi':>11} "
              f"{'rel_err_lapse':>13} {'chi_spread':>10} {'Ham_rms/K^2':>11} "
              f"{'Mom_max':>9}")
        max_err = {"K": 0.0, "chi": 0.0, "lapse": 0.0}
        for row in rows:
            exact = analytic(row["t"], K0, p)
            err = {
                "K": row["K_mean"] / exact["K"] - 1.0,
                "chi": row["chi_mean"] / exact["chi"] - 1.0,
                "lapse": row["lapse_mean"] / exact["lapse"] - 1.0,
            }
            for key in max_err:
                max_err[key] = max(max_err[key], abs(err[key]))
            spread = row["chi_max"] - row["chi_min"]
            # Ham ~ (2/3) K^2 - 2 rho with each term ~ (2/3) K^2
            ham_rel = row["Ham_rms"] / (2.0 / 3.0 * row["K_mean"] ** 2)
            if row["step"] % 5 == 0 or row is rows[-1]:
                print(f"{row['t']:8.4f} {row['K_mean'] / K0:10.4f} "
                      f"{err['K']:10.2e} {err['chi']:11.2e} "
                      f"{err['lapse']:13.2e} {spread:10.2e} {ham_rel:11.2e} "
                      f"{row['Mom_max']:9.1e}")
        print("max |relative error|: " +
              ", ".join(f"{key} {value:.3e}" for key, value in max_err.items()))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()
