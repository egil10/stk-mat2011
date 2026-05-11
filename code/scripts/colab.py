"""
One-call Colab setup for stk-mat2011 notebooks.

In your first notebook cell, just two lines:

    !curl -sL https://raw.githubusercontent.com/egil10/stk-mat2011/main/code/scripts/colab.py -o /content/colab.py
    import sys; sys.path.insert(0, '/content'); from colab import setup; setup('code/jan2025')

`setup(notebook_dir)` mounts Drive, clones (or pulls) the repo, symlinks
`code/data/processed` to the data folder in Drive, adds `code/scripts`
to `sys.path`, and `chdir`s into the requested notebook directory so
relative paths (`../data/processed`, etc.) keep working.
"""
import os
import shutil
import subprocess
import sys


REPO_ROOT  = '/content/stk-mat2011'
REPO_URL   = 'https://github.com/egil10/stk-mat2011.git'
DRIVE_ROOT = '/content/drive/MyDrive/GITHUB-COPILOT/stk-mat2011'
REPO_DATA  = f'{REPO_ROOT}/code/data/processed'
DRIVE_DATA = f'{DRIVE_ROOT}/data/processed'


def _mount_drive():
    """Mount /content/drive (only works inside Colab)."""
    from google.colab import drive
    drive.mount('/content/drive')


def _clone_or_pull():
    """Clone the repo on first run, pull on subsequent runs."""
    if not os.path.isdir(REPO_ROOT):
        subprocess.run(['git', 'clone', REPO_URL, REPO_ROOT], check=True)
    else:
        subprocess.run(['git', '-C', REPO_ROOT, 'pull'], check=True)


# Names of project modules whose cached versions in sys.modules need to be
# invalidated after a git pull, so that the next `import` picks up the fresh
# source.  Without this, re-running the bootstrap cell silently keeps using
# the previous run's stale code (Python only loads each module once).
_PROJECT_MODULES = {
    'month', 'spread', 'engine', 'backtester', 'tearsheet',
    'screener', 'plotting', 'descriptive', 'wfo', 'synthetic',
}


def _invalidate_project_modules():
    """Drop any cached project modules so the next import re-reads from disk."""
    for mod in list(sys.modules):
        if mod in _PROJECT_MODULES:
            del sys.modules[mod]


def _symlink_drive_data():
    """Replace the empty data/processed in the clone with a Drive symlink."""
    if os.path.isdir(REPO_DATA) and not os.path.islink(REPO_DATA):
        shutil.rmtree(REPO_DATA)
    if os.path.islink(REPO_DATA) and not os.path.exists(REPO_DATA):
        os.unlink(REPO_DATA)
    if not os.path.islink(REPO_DATA):
        os.symlink(DRIVE_DATA, REPO_DATA)


def setup(notebook_dir, verbose=True):
    """
    Mount Drive, clone or pull the repo, symlink the data folder, add
    scripts/ to sys.path, and chdir into the notebook's repo-relative folder.

    Parameters
    ----------
    notebook_dir : str
        Repo-relative path, e.g. 'code/jan2025' or 'code/notebooks'.
    verbose : bool
        Print a small sanity summary after setup.
    """
    _mount_drive()
    _clone_or_pull()
    _symlink_drive_data()
    _invalidate_project_modules()

    scripts_path = f'{REPO_ROOT}/code/scripts'
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)

    target = f'{REPO_ROOT}/{notebook_dir.strip("/")}'
    if not os.path.isdir(target):
        raise FileNotFoundError(
            f"notebook_dir {target!r} does not exist inside the cloned repo."
        )
    os.chdir(target)

    if verbose:
        n_parquet = len([f for f in os.listdir(REPO_DATA) if f.endswith('.parquet')])
        link_target = os.readlink(REPO_DATA) if os.path.islink(REPO_DATA) else 'N/A'
        print(f"CWD           : {os.getcwd()}")
        print(f"Scripts path  : {scripts_path}")
        print(f"Data symlink  : {REPO_DATA} -> {link_target}")
        print(f"Parquet files : {n_parquet}")
