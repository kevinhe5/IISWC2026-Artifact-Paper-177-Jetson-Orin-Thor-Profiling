#!/usr/bin/env python3
"""Generate fig_O_agentic_react_cache.{pdf,png}
   — Figure O: Agentic AI ReAct trace, TTFT vs turn, cache ON vs OFF.

Workload (Methodology A): 10 synthetic ReAct traces × 30 turns; shared
~1.5 K-tok system prompt (ReAct preamble + 20 tool schemas), ~400-tok
tool observations per turn → prompt grows to ~17 K tok by turn 29.
Mirrors how Devin / OpenHands / SWEAgent fire prompts: same long
system prompt every iteration, growing trace history.

Layout: single panel with twin y-axes; speedup ratios called out as
a text-box note inside the plot.

  Left y  (log) : TTFT per turn (median over 10 dialogues, IQR shaded),
                  cache OFF (dashed square) vs cache ON (solid circle).
  Right y (lin) : Per-turn cache speedup ratio = TTFT_off / TTFT_on,
                  one dotted-diamond line per framework.
  Note panel    : aggregate-trace cache speedup per framework.

Llama-3.2-1B fp16, AGX Orin 32 GB locked clocks.

================================================================================
Self-contained data
================================================================================
DATA below is the per-turn aggregated TTFT (median + IQR over the 10
dialogues) for each framework × cache state.  Extracted from the
the 30-turn ReAct CSVs:
  /nvme/ispass/jetson-containers/data/benchmarks/sweep_results/
    vllm_agent_20260505_174843.csv      (vLLM)
    sglang_agent_20260505_183911.csv    (SGLang)
    llamacpp_agent_20260506_025851.csv  (llama.cpp)

The llama.cpp slot-promotion shortcut occasionally evaluates only a
handful of new delta tokens between turns (turns 1, 5, 8 in this
trace), reporting sub-50 ms cache-ON TTFTs that produce
physically-implausible speedup ratios.  Those three turns are dropped
from llama.cpp's cache-ON series at extraction time (filter
MIN_CACHED_TTFT_MS = 50.0 in the original loader).  No filter is
applied to vLLM or SGLang.

================================================================================
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PDF = "/nvme/ispass/paper_jetson/JetsonAnalysis/figs/fig_O_agentic_react_cache.pdf"
OUT_PNG = "/nvme/ispass/paper_jetson/JetsonAnalysis/figs/fig_O_agentic_react_cache.png"

# (fw_label, color) — top-to-bottom by aggregate speedup, so legend &
# note box read llama.cpp / SGLang / vLLM.
FW_ORDER = ["llama.cpp", "SGLang", "vLLM"]
FW_COLOR = {
    "llama.cpp": "#f472b6",
    "SGLang":    "#a78bfa",
    "vLLM":      "#f97316",
}

# Per-turn aggregated stats across 10 dialogues.  Each tuple is
# (turn_idx, median_ttft_ms, q25_ttft_ms, q75_ttft_ms).
# llama.cpp cache-ON turns 1 / 5 / 8 are intentionally absent
# (slot-promotion shortcut artifact; see docstring).
DATA = {
    "llama.cpp": {
        "on":  [
            ( 0,     325.56,     324.66,     326.12),
            ( 2,     255.74,     254.97,     258.58),
            ( 3,     134.17,     132.72,     134.63),
            ( 4,      72.78,      70.62,      73.67),
            ( 6,      97.06,      95.83,      97.76),
            ( 7,     410.13,     409.78,     410.93),
            ( 9,     463.66,     462.85,     464.28),
            (10,     333.95,     333.14,     337.21),
            (11,     228.24,     225.76,     229.16),
            (12,     113.09,     111.90,     113.89),
            (13,     248.19,     246.89,     249.17),
            (14,     617.94,     617.27,     618.90),
            (15,     130.01,     129.02,     130.20),
            (16,     683.44,     681.75,     684.01),
            (17,     545.75,     542.66,     550.21),
            (18,     379.31,     376.00,     384.12),
            (19,     215.06,     214.36,     215.47),
            (20,     394.24,     391.02,     395.42),
            (21,     855.45,     851.69,     855.96),
            (22,     222.61,     221.42,     223.32),
            (23,     933.56,     932.68,     934.95),
            (24,     801.70,     794.49,     817.03),
            (25,     584.67,     571.62,     588.55),
            (26,     325.76,     323.85,     326.72),
            (27,     596.74,     592.50,     597.63),
            (28,    1797.22,    1795.12,    1798.21),
            (29,     322.22,     320.33,     324.13),
        ],
        "off":  [
            ( 0,     325.23,     324.95,     325.85),
            ( 1,     482.94,     482.67,     484.21),
            ( 2,     916.33,     916.03,     916.60),
            ( 3,    1176.41,    1176.23,    1176.97),
            ( 4,    1468.28,    1468.16,    1468.49),
            ( 5,    1750.20,    1749.87,    1750.92),
            ( 6,    2095.81,    2095.34,    2096.53),
            ( 7,    2537.20,    2534.62,    2543.37),
            ( 8,    2862.73,    2861.86,    2863.16),
            ( 9,    3489.73,    3487.76,    3496.27),
            (10,    4201.41,    4200.39,    4202.09),
            (11,    4701.81,    4700.89,    4702.09),
            (12,    4988.40,    4986.01,    4991.05),
            (13,    5523.46,    5519.33,    5524.86),
            (14,    6360.80,    6359.84,    6363.47),
            (15,    6773.28,    6765.70,    6774.72),
            (16,    7710.10,    7702.98,    7711.37),
            (17,    8573.30,    8566.09,    8581.53),
            (18,    9279.95,    9269.19,    9293.49),
            (19,    9845.51,    9830.65,    9855.46),
            (20,   10573.02,   10565.58,   10578.38),
            (21,   11689.66,   11685.03,   11697.16),
            (22,   12310.43,   12276.45,   12320.36),
            (23,   13629.45,   13601.39,   13762.69),
            (24,   14812.31,   14786.29,   14824.95),
            (25,   15816.69,   15803.94,   15829.74),
            (26,   16803.06,   16801.21,   16809.58),
            (27,   18103.31,   18043.48,   18118.90),
            (28,   19719.13,   19692.28,   19761.10),
            (29,   21363.53,   21357.05,   21390.04),
        ],
    },
    "SGLang": {
        "on":  [
            ( 0,      54.66,      54.27,      56.93),
            ( 1,      65.70,      65.14,      66.26),
            ( 2,     110.53,     110.16,     111.09),
            ( 3,     113.27,     112.38,     115.16),
            ( 4,      92.37,      92.14,      92.73),
            ( 5,      95.08,      94.89,      95.42),
            ( 6,     100.13,      99.83,     100.47),
            ( 7,     142.55,     141.98,     142.98),
            ( 8,      94.54,      94.24,      94.60),
            ( 9,     159.23,     158.66,     160.11),
            (10,     161.44,     160.56,     161.77),
            (11,     127.84,     127.54,     128.57),
            (12,     130.00,     129.91,     130.22),
            (13,     136.02,     135.47,     136.97),
            (14,     190.86,     190.00,     191.94),
            (15,     125.44,     125.03,     126.08),
            (16,     205.86,     205.12,     206.37),
            (17,     210.32,     209.59,     212.00),
            (18,     163.68,     162.79,     165.67),
            (19,     166.12,     166.00,     167.36),
            (20,     171.43,     170.86,     171.52),
            (21,     235.44,     234.90,     235.75),
            (22,     154.02,     153.62,     154.39),
            (23,     252.25,     251.16,     253.11),
            (24,     256.02,     254.38,     257.22),
            (25,     198.34,     197.55,     202.06),
            (26,     201.11,     199.88,     201.66),
            (27,     206.84,     205.86,     207.21),
            (28,     283.22,     282.57,     285.57),
            (29,     185.65,     184.25,     186.20),
        ],
        "off":  [
            ( 0,     156.34,     155.69,     156.80),
            ( 1,     209.45,     209.42,     209.89),
            ( 2,     312.22,     311.97,     312.61),
            ( 3,     394.43,     393.94,     394.99),
            ( 4,     461.84,     460.88,     462.89),
            ( 5,     525.95,     525.51,     526.85),
            ( 6,     606.14,     605.65,     606.65),
            ( 7,     726.64,     725.97,     727.80),
            ( 8,     798.20,     797.63,     798.87),
            ( 9,     915.23,     915.03,     917.11),
            (10,    1044.01,    1042.73,    1044.36),
            (11,    1154.28,    1153.91,    1154.78),
            (12,    1217.07,    1216.49,    1217.79),
            (13,    1307.05,    1306.67,    1308.41),
            (14,    1424.86,    1423.52,    1425.99),
            (15,    1497.11,    1495.89,    1499.51),
            (16,    1668.12,    1666.47,    1669.45),
            (17,    1817.14,    1816.33,    1818.55),
            (18,    1933.47,    1932.31,    1934.12),
            (19,    2021.66,    2020.77,    2022.29),
            (20,    2154.97,    2153.45,    2156.35),
            (21,    2334.01,    2310.90,    2334.67),
            (22,    2414.14,    2412.43,    2433.58),
            (23,    2604.23,    2603.25,    2611.15),
            (24,    2788.24,    2786.47,    2812.17),
            (25,    2929.94,    2928.01,    2931.91),
            (26,    3081.23,    3078.93,    3082.85),
            (27,    3180.52,    3177.90,    3181.47),
            (28,    3395.84,    3393.64,    3397.45),
            (29,    3502.29,    3500.53,    3503.38),
        ],
    },
    "vLLM": {
        "on":  [
            ( 0,      29.73,      29.52,      29.92),
            ( 1,      60.46,      60.06,      60.70),
            ( 2,     119.32,     118.95,     119.92),
            ( 3,     102.33,     102.25,     102.79),
            ( 4,      90.30,      80.41,      90.57),
            ( 5,      78.52,      78.07,      78.83),
            ( 6,      85.64,      85.29,      85.84),
            ( 7,     115.55,     115.09,     116.58),
            ( 8,      86.62,      81.78,      86.89),
            ( 9,     139.14,     130.31,     147.51),
            (10,     128.52,     128.28,     128.69),
            (11,     113.15,     104.09,     113.33),
            (12,      97.56,      97.46,      98.18),
            (13,     105.93,     105.47,     106.53),
            (14,     143.17,     142.00,     143.57),
            (15,     102.35,      98.15,     106.82),
            (16,     176.24,     175.66,     176.76),
            (17,     153.66,     153.52,     153.97),
            (18,     130.25,     124.20,     136.43),
            (19,     118.30,     118.26,     118.62),
            (20,     128.24,     127.72,     129.02),
            (21,     169.53,     168.88,     170.62),
            (22,     118.06,     116.72,     125.69),
            (23,     203.30,     188.20,     203.93),
            (24,     178.81,     178.65,     179.22),
            (25,     158.26,     144.14,     159.06),
            (26,     137.70,     137.34,     138.67),
            (27,     148.75,     148.51,     149.39),
            (28,     198.50,     195.36,     199.01),
            (29,     145.43,     138.06,     146.51),
        ],
        "off":  [
            ( 0,     162.60,     162.43,     162.78),
            ( 1,     207.75,     206.39,     208.38),
            ( 2,     295.68,     294.96,     297.17),
            ( 3,     368.48,     367.82,     369.18),
            ( 4,     426.69,     425.50,     429.56),
            ( 5,     476.71,     476.19,     479.48),
            ( 6,     542.54,     539.56,     543.58),
            ( 7,     631.16,     630.31,     632.75),
            ( 8,     683.31,     682.83,     684.54),
            ( 9,     770.42,     769.69,     772.03),
            (10,     868.16,     867.69,     869.80),
            (11,     941.96,     940.84,     943.30),
            (12,     978.13,     977.07,     979.75),
            (13,    1062.00,    1061.06,    1063.45),
            (14,    1162.34,    1160.38,    1162.99),
            (15,    1213.12,    1212.24,    1215.12),
            (16,    1337.53,    1336.83,    1339.12),
            (17,    1433.88,    1433.54,    1435.52),
            (18,    1514.80,    1514.38,    1515.25),
            (19,    1578.66,    1577.54,    1581.10),
            (20,    1665.18,    1663.55,    1666.95),
            (21,    1777.82,    1775.01,    1778.99),
            (22,    1838.78,    1836.53,    1841.61),
            (23,    1959.64,    1957.69,    1966.41),
            (24,    2084.12,    2083.99,    2089.77),
            (25,    2179.70,    2179.03,    2183.20),
            (26,    2262.95,    2261.97,    2264.68),
            (27,    2326.38,    2325.51,    2329.60),
            (28,    2472.99,    2471.96,    2477.07),
            (29,    2554.61,    2548.22,    2555.95),
        ],
    },
}


def unzip(stats):
    """Convert list of (turn, med, q25, q75) tuples into 4 numpy arrays."""
    if not stats:
        return tuple(np.array([]) for _ in range(4))
    a = np.array(stats, dtype=float)
    return a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3]


def main():
    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    summary = []
    for label in FW_ORDER:
        color = FW_COLOR[label]
        t_on,  m_on,  qL_on,  qH_on  = unzip(DATA[label]["on"])
        t_off, m_off, qL_off, qH_off = unzip(DATA[label]["off"])

        # TTFT — left axis (log)
        ax.fill_between(t_off, qL_off, qH_off, color=color, alpha=0.10, zorder=2)
        ax.plot(t_off, m_off, ls="--", color=color, lw=1.8, marker="s",
                ms=4.5, mec="#0a0a0e", mew=0.4, zorder=4,
                label=f"{label}  cache OFF")
        ax.fill_between(t_on, qL_on, qH_on, color=color, alpha=0.18, zorder=2)
        ax.plot(t_on, m_on, ls="-", color=color, lw=1.8, marker="o",
                ms=4.5, mec="#0a0a0e", mew=0.4, zorder=4,
                label=f"{label}  cache ON")

        agg_on  = float(np.sum(m_on))
        agg_off = float(np.sum(m_off))
        summary.append((label, color,
                        m_on[0], m_off[0], m_on[-1], m_off[-1],
                        agg_on, agg_off, agg_off / max(agg_on, 1e-6)))

    # Left axis (TTFT) cosmetics
    ax.set_yscale("log")
    ax.set_xlim(-1, 30)
    all_y = []
    for s in summary:
        all_y.extend([s[2], s[3], s[4], s[5]])
    ax.set_ylim(min(all_y) * 0.6, max(all_y) * 2.5)
    ax.set_xlabel("turn index", fontsize=14)
    ax.set_ylabel("TTFT (ms)", fontsize=14)
    ax.grid(True, which="both", color="#e5e7eb", lw=0.5, alpha=0.55, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top",):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

    # Legend ABOVE the plot — 6 TTFT entries (cache ON/OFF per framework).
    h1, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h1,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, fontsize=10, frameon=False,
              handlelength=2.4, handleheight=1.0,
              columnspacing=1.4, labelspacing=0.4)

    # Note box — aggregate cache speedup per framework, upper-left
    # of the plot (empty headroom above all TTFT curves).
    note_lines = ["Aggregate cache speedup"]
    for s in sorted(summary, key=lambda r: -r[8]):
        note_lines.append(f"  {s[0]}: {s[8]:.1f}×")
    note_text = "\n".join(note_lines)
    ax.text(0.015, 0.97, note_text,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=17, fontweight="bold", color="#1f2937",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white", edgecolor="#d1d5db",
                      lw=0.8, alpha=0.95))

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.18, dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print()
    print("=== summary ===")
    print(f"  {'fw':<10}  {'t0_on':>8} {'t0_off':>8}  {'t29_on':>8} {'t29_off':>8}  "
          f"{'agg_on':>9} {'agg_off':>9}  {'ratio':>6}")
    for s in summary:
        label, _color, t0o, t0f, tNo, tNf, ao, af, r = s
        print(f"  {label:<10}  {t0o:>8.1f} {t0f:>8.1f}  {tNo:>8.1f} {tNf:>8.1f}  "
              f"{ao/1000:>8.2f}s {af/1000:>8.2f}s  {r:>5.2f}×")


if __name__ == "__main__":
    main()
