"""Block diagram of the combined-cycle plant with the state numbering used throughout the code."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
GAS = "#c0392b"
WATER = "#1f5fa8"
SHAFT = "0.25"


def _path(ax, points, color, lw=2.2):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="round")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "mutation_scale": 16})


def _state(ax, xy, text, color):
    ax.text(*xy, text, ha="center", va="center", fontsize=12, fontweight="bold", color=color,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "lw": 1.2}, zorder=10)


def draw_schematic(fig_name="schematic.png"):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    face = {"fc": "#f4f6f8", "ec": "0.2", "lw": 1.6}

    ax.add_patch(Polygon([(1.6, 6.0), (1.6, 7.8), (3.4, 7.3), (3.4, 6.5)], **face))
    ax.text(2.45, 6.9, "Compressor", ha="center", va="center", fontsize=11)
    ax.add_patch(Polygon([(7.4, 6.5), (7.4, 7.3), (9.2, 7.8), (9.2, 6.0)], **face))
    ax.text(8.35, 6.9, "Gas\nturbine", ha="center", va="center", fontsize=11)
    ax.add_patch(Rectangle((4.5, 7.75), 1.8, 0.75, **face))
    ax.text(5.4, 8.12, "Combustor", ha="center", va="center", fontsize=11)
    ax.plot([3.4, 7.4], [6.9, 6.9], color=SHAFT, lw=3)
    ax.plot([5.4, 5.4], [6.9, 6.15], color=SHAFT, lw=3)
    ax.add_patch(Circle((5.4, 5.8), 0.35, **face))
    ax.text(5.4, 5.8, "G", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.annotate("", xy=(6.6, 5.8), xytext=(5.75, 5.8),
                arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 2, "mutation_scale": 16})
    ax.text(6.7, 5.8, "$\\dot W_{GT}$", ha="left", va="center", fontsize=14)

    _path(ax, [(0.3, 6.9), (1.6, 6.9)], GAS)
    ax.text(0.3, 6.35, "air, 100 kg/s\n298 K, 1.013 bar", fontsize=9, va="top")
    _state(ax, (0.9, 7.25), "1", GAS)
    _path(ax, [(3.4, 7.2), (3.9, 8.12), (4.5, 8.12)], GAS)
    _state(ax, (3.95, 7.55), "2", GAS)
    _path(ax, [(6.3, 8.12), (6.9, 8.12), (7.4, 7.2)], GAS)
    _state(ax, (7.0, 7.55), "3", GAS)
    _path(ax, [(5.4, 8.95), (5.4, 8.5)], "0.35")
    ax.text(5.55, 8.78, "fuel CH$_4$ ($\\dot m_f$, LHV)", fontsize=10, va="center")

    hx0, hx1, gas_x, water_x = 10.4, 12.6, 10.95, 12.05
    sections = [(6.0, 8.0, "Superheater"), (4.0, 6.0, "Evaporator\n(drum)"), (2.0, 4.0, "Economizer")]
    for y0, y1, name in sections:
        ax.add_patch(Rectangle((hx0, y0), hx1 - hx0, y1 - y0, fc="#fbf3e6", ec="0.2", lw=1.6))
        ax.text((hx0 + hx1) / 2, (y0 + y1) / 2, name, ha="center", va="center", fontsize=10, rotation=90)
    ax.text((hx0 + hx1) / 2, 8.25, "HRSG", ha="center", va="bottom", fontsize=12, fontweight="bold")

    _path(ax, [(9.2, 6.9), (9.8, 6.9), (9.8, 8.6), (gas_x, 8.6), (gas_x, 8.0)], GAS)
    _state(ax, (9.8, 7.75), "4", GAS)
    for y_top, y_bot in ((8.0, 6.0), (6.0, 4.0), (4.0, 2.0)):
        _path(ax, [(gas_x, y_top - 0.15), (gas_x, y_bot + 0.15)], GAS, lw=1.8)
    _state(ax, (gas_x, 6.0), "4a", GAS)
    _state(ax, (gas_x, 4.0), "4b", GAS)
    _path(ax, [(gas_x, 2.0), (gas_x, 1.2), (9.9, 1.2)], GAS)
    _state(ax, (gas_x, 1.45), "4c", GAS)
    ax.add_patch(Rectangle((9.1, 0.6), 0.8, 2.6, fc="0.85", ec="0.2", lw=1.6))
    ax.text(9.5, 1.9, "Stack", ha="center", va="center", fontsize=10, rotation=90)
    _path(ax, [(9.5, 3.2), (9.5, 4.0)], GAS, lw=1.6)
    ax.text(9.5, 4.1, "to atmosphere", ha="center", va="bottom", fontsize=9)

    for y_bot, y_top in ((2.0, 4.0), (4.0, 6.0), (6.0, 8.0)):
        _path(ax, [(water_x, y_bot + 0.15), (water_x, y_top - 0.15)], WATER, lw=1.8)
    _state(ax, (water_x, 4.0), "8a", WATER)
    _state(ax, (water_x, 6.0), "8b", WATER)

    ax.add_patch(Polygon([(13.3, 6.2), (13.3, 7.0), (15.0, 7.5), (15.0, 5.7)], **face))
    ax.text(14.15, 6.6, "Steam\nturbine", ha="center", va="center", fontsize=11)
    _path(ax, [(water_x, 8.0), (water_x, 8.6), (12.9, 8.6), (12.9, 6.6), (13.3, 6.6)], WATER)
    _state(ax, (12.9, 7.8), "5", WATER)
    ax.plot([15.0, 15.25], [6.6, 6.6], color=SHAFT, lw=3)
    ax.add_patch(Circle((15.55, 6.6), 0.3, **face))
    ax.text(15.55, 6.6, "G", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.annotate("", xy=(15.55, 7.65), xytext=(15.55, 6.9),
                arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 2, "mutation_scale": 16})
    ax.text(15.55, 7.75, "$\\dot W_{ST}$", ha="center", va="bottom", fontsize=14)

    ax.add_patch(Rectangle((13.4, 3.1), 2.0, 1.2, **face))
    ax.text(14.4, 3.7, "Condenser", ha="center", va="center", fontsize=11)
    _path(ax, [(14.6, 5.8), (14.6, 4.3)], WATER)
    _state(ax, (14.95, 5.05), "6", WATER)
    ax.annotate("", xy=(15.85, 3.45), xytext=(15.4, 3.45),
                arrowprops={"arrowstyle": "-|>", "color": "0.45", "lw": 1.5})
    ax.annotate("", xy=(15.4, 3.95), xytext=(15.85, 3.95),
                arrowprops={"arrowstyle": "-|>", "color": "0.45", "lw": 1.5})
    ax.text(15.9, 3.7, "cooling\nwater", fontsize=8, va="center", ha="left", color="0.35")

    ax.add_patch(Circle((14.4, 1.2), 0.38, **face))
    ax.text(14.4, 1.2, "Pump", ha="center", va="center", fontsize=9)
    _path(ax, [(14.4, 3.1), (14.4, 1.58)], WATER)
    _state(ax, (14.75, 2.35), "7", WATER)
    _path(ax, [(14.02, 1.2), (water_x, 1.2), (water_x, 2.0)], WATER)
    _state(ax, (13.1, 1.2), "8", WATER)
    ax.annotate("", xy=(14.4, 0.82), xytext=(14.4, 0.25),
                arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 1.5})
    ax.text(14.55, 0.35, "$\\dot W_{pump}$", fontsize=11, va="center")

    ax.plot([], [], color=GAS, lw=2.5, label="air / combustion gas")
    ax.plot([], [], color=WATER, lw=2.5, label="water / steam")
    ax.plot([], [], color=SHAFT, lw=3, label="shaft")
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.set_title("Methane-fired combined cycle: Brayton topping cycle, single-pressure HRSG, Rankine bottoming cycle",
                 fontsize=14)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    draw_schematic()
