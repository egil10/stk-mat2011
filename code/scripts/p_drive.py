import os
import shutil
from pathlib import Path

def mount_drive():
    """Mounts Google Drive if running in Colab."""
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        return True
    except ImportError:
        print("Not running in Colab. Skipping Drive mount.")
        return False

def sync_data(src_path, drive_subdir="GITHUB-COLAB/stk-mat2011/data", to_drive=True):
    """
    Syncs a file or folder between Colab and Google Drive.
    
    Args:
        src_path (str or Path): Local path in Colab.
        drive_subdir (str): Folder in Google Drive (under MyDrive).
        to_drive (bool): If True, copies to Drive. If False, copies from Drive to Local.
    """
    local_path = Path(src_path)
    drive_base = Path("/content/drive/MyDrive") / drive_subdir
    drive_path = drive_base / local_path.name
    
    # Ensure drive folder exists
    if to_drive:
        drive_base.mkdir(parents=True, exist_ok=True)
        if local_path.is_dir():
            if drive_path.exists(): shutil.rmtree(drive_path)
            shutil.copytree(local_path, drive_path)
        else:
            shutil.copy2(local_path, drive_path)
        print(f"Backed up {local_path.name} to Google Drive ✅")
    else:
        # Syncing from Drive to Local
        if not drive_path.exists():
            print(f"ERROR: Could not find {local_path.name} on Drive at {drive_path}")
            return
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if drive_path.is_dir():
            if local_path.exists(): shutil.rmtree(local_path)
            shutil.copytree(drive_path, local_path)
        else:
            shutil.copy2(drive_path, local_path)
        print(f"Loaded {local_path.name} from Google Drive to Local ✅")
