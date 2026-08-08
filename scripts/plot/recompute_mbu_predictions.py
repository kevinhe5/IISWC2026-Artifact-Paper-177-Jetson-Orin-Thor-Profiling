#!/usr/bin/env python3
"""Recompute Table 3 (b) MBU predictions at a chosen sustained-BW ratio S_BW.

Eq. 3 from the paper:
    MBU_pred = (m0/S_BW) / (m0/S_BW + (1-m0)/R)
where
    R  = (rho_c/S_c + rho_pi/S_pi + rho_l/S_l)^-1              (tier-decomp)
    m0 = per-framework baseline MBU, backed out of the published m_pred at
         S_BW_old assuming the same R.

Only S_BW changes across paper revisions; R, rho, m0 (per-framework) are fixed.
The published Table 3 (b) uses S_BW = 1.37 (Thor STREAM-triad sustained ratio).

Usage:
    python3 recompute_mbu_predictions.py                       # default S_BW_new=1.37
    python3 recompute_mbu_predictions.py --s-bw-new 1.28       # e.g. earlier revision
    python3 recompute_mbu_predictions.py --s-bw-new 1.37 --csv # emit CSV rows

Outputs the framework-by-framework prediction, |Δ| vs measured, and overall MAE.
"""
import argparse

# Tier-decomposition constants (paper Table 3 (a))
S_c, S_pi, S_l = 4.89, 14.83, 1.66

# Published m_pred at S_BW=1.18 (paper original); m_meas is fixed by measurement.
# (framework, tier, (rho_c, rho_pi, rho_l), m_pred_old_%, m_meas_%)
FW = [
    ("trtedge_llm",     1, (.40, .10, .50), 86.6, 84.5),
    ("llamacpp",        1, (.40, .10, .50), 74.9, 81.8),
    ("vllm",            2, (.10, .40, .50), 78.1, 70.2),
    ("sglang",          2, (.10, .40, .50), 72.6, 71.8),
    ("pytorch_compile", 2, (.10, .40, .50), 76.5, 67.0),
    ("pytorch_eager",   3, (.10, .70, .20), 68.0, 57.2),
]
S_BW_OLD = 1.18  # published m_pred_old baseline


def R_of(rho):
    rc, rp, rl = rho
    return 1.0 / (rc / S_c + rp / S_pi + rl / S_l)


def mbu(m0, s_bw, R):
    a = m0 / s_bw
    b = (1.0 - m0) / R
    return a / (a + b)


def invert_m0(m_pred, s_bw, R):
    """m0 = p*S_BW / [(1-p)*R + p*S_BW]"""
    p = m_pred
    return p * s_bw / ((1 - p) * R + p * s_bw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s-bw-new", type=float, default=1.37,
                    help="Sustained-BW ratio to recompute at (default 1.37 = published Tab 3(b)).")
    ap.add_argument("--csv", action="store_true",
                    help="Emit CSV rows matching data/mbu/predictions_thor.csv schema.")
    args = ap.parse_args()

    s_new = args.s_bw_new
    deltas = []
    rows = []
    for name, tier, rho, mp_old, mmeas in FW:
        R = R_of(rho)
        m0 = invert_m0(mp_old / 100.0, S_BW_OLD, R)
        mp_new = mbu(m0, s_new, R) * 100.0
        d = mp_new - mmeas
        deltas.append(d)
        rows.append((name, tier, rho, R, m0 * 100.0, mp_old, mp_new, mmeas, d))

    if args.csv:
        print("framework,tier,rho_c,rho_pi,rho_l,mbu_pred_pct,mbu_meas_pct,abs_err_pp,"
              f"mbu_pred_sustained{int(round(s_new*100))}_pct,abs_err_sustained_pp")
        for name, tier, rho, R, m0, mp_old, mp_new, mmeas, d in rows:
            print(f"{name},{tier},{rho[0]},{rho[1]},{rho[2]},"
                  f"{mp_old:.1f},{mmeas:.1f},{abs(mp_old-mmeas):.1f},"
                  f"{mp_new:.1f},{abs(d):.1f}")
        return

    print(f"{'framework':16} {'R':>5} {'m0(back-out)':>13} {'pred_1.18':>9} "
          f"{'pred_'+format(s_new,'.2f'):>9} {'meas':>6} {'delta':>6}")
    for name, tier, rho, R, m0, mp_old, mp_new, mmeas, d in rows:
        print(f"{name:16} {R:5.3f} {m0:12.1f}% {mp_old:8.1f}% "
              f"{mp_new:8.1f}% {mmeas:5.1f}% {d:+6.1f}")

    mae = sum(abs(v) for v in deltas) / len(deltas)
    bias = sum(deltas) / len(deltas)
    print(f"\n  S_BW={s_new:.2f} : MAE={mae:.2f} pp   bias(pred-meas)={bias:+.2f} pp")
    print("  R by tier:  T1={:.3f}  T2={:.3f}  T3={:.3f}".format(
        R_of((.40, .10, .50)), R_of((.10, .40, .50)), R_of((.10, .70, .20))))


if __name__ == "__main__":
    main()
