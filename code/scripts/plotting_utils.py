from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/GITHUB-COPILOT/stk-mat2011")
DEFAULT_PDF_DPI = 180


# ── Economist-inspired palette ──────────────────────────────────────
ECON = {
    # Strategy colours
    'BuyHold':  '#F4A261',   # warm muted orange
    'Baseline': '#6C757D',   # neutral grey
    'AR':       '#0077B6',   # clean medium blue
    'MS_AR':    '#9D4EDD',   # restrained purple

    # General purpose
    'navy':     '#0B1F33',
    'teal':     '#005F73',
    'blue':     '#0077B6',
    'blue_grey':'#4A90A4',
    'red':      '#D62828',
    'deep_red': '#B22222',
    'coral':    '#E85D5A',
    'orange':   '#F4A261',
    'green':    '#2A9D8F',
    'olive':    '#6A994E',
    'purple':   '#9D4EDD',
    'deep_purp':'#7B2CBF',
    'brown':    '#8D6E63',
    'grey':     '#6C757D',
    'lt_grey':  '#ADB5BD',
    'grid':     '#E9ECEF',
    'cream':    '#F7F3E8',
    'paper':    '#FBFAF7',
}

# Ordered list for cycling through colours
ECON_CYCLE = [
    ECON['navy'], ECON['blue'], ECON['red'], ECON['orange'],
    ECON['green'], ECON['purple'], ECON['teal'], ECON['coral'],
    ECON['olive'], ECON['brown'], ECON['blue_grey'], ECON['deep_red'],
]

# Two-asset pair (A vs B)
COL_A = ECON['blue']
COL_B = ECON['red']


def apply_econ_style():
    """Apply Economist-inspired matplotlib defaults globally."""
    plt.rcParams.update({
        'figure.facecolor':  ECON['paper'],
        'axes.facecolor':    ECON['paper'],
        'axes.edgecolor':    ECON['lt_grey'],
        'axes.grid':         True,
        'grid.color':        ECON['grid'],
        'grid.alpha':        0.6,
        'axes.prop_cycle':   plt.cycler(color=ECON_CYCLE),
        'axes.titleweight':  'bold',
        'axes.titlesize':    12,
        'axes.labelsize':    10,
        'xtick.color':       ECON['grey'],
        'ytick.color':       ECON['grey'],
        'text.color':        ECON['navy'],
        'font.family':       'sans-serif',
        'legend.framealpha':  0.9,
        'legend.edgecolor':  ECON['grid'],
    })


def default_pdf_dir():
    """Use the shared Drive folder in Colab, otherwise fall back to a local folder."""
    if DEFAULT_DRIVE_ROOT.exists():
        return DEFAULT_DRIVE_ROOT / "plots"
    return Path("plots")


def pdf_filename(prefix, plot_name):
    safe_plot_name = str(plot_name).strip().replace(" ", "_")
    safe_prefix = str(prefix).strip().replace(" ", "_") if prefix else ""
    if safe_prefix:
        return f"{safe_prefix}_{safe_plot_name}.pdf"
    return f"{safe_plot_name}.pdf"


def rasterize_heavy_artists(fig):
    """Keep text/vector axes sharp while making dense plot data cheap for LaTeX."""
    for ax in fig.axes:
        for artist in [*ax.lines, *ax.collections]:
            artist.set_rasterized(True)


def save_figure_pdf(
    fig,
    filename,
    pdf_dir=None,
    enabled=False,
    rasterize=True,
    dpi=DEFAULT_PDF_DPI,
):
    if not enabled:
        return None

    output_dir = Path(pdf_dir) if pdf_dir is not None else default_pdf_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")

    if rasterize:
        rasterize_heavy_artists(fig)

    with mpl.rc_context({
        "pdf.compression": 9,
        "path.simplify": True,
        "path.simplify_threshold": 1.0,
        "agg.path.chunksize": 10000,
    }):
        fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=dpi)

    print(f"Saved plot PDF: {output_path}")
    return output_path
