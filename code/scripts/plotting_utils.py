from pathlib import Path

import matplotlib as mpl


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/GITHUB-COPILOT/stk-mat2011")
DEFAULT_PDF_DPI = 180


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
