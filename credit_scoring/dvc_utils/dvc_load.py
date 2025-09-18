import os

from pathlib import Path

def download_dvc_data(
    dvc_files_path: str,
    remote_name: str = None
):
    """
    Downloads data from DVC remote storage using CLI commands.
    """
    try:
        print(f"Downloading data from DVC remote to {dvc_files_path}...")

        os.system(f"cd {dvc_files_path}")
        cmd = 'dvc pull'

        if remote_name:
            cmd += f"-r {remote_name}"

        os.system(cmd)
        os.system("cd ../credit_scoring")

    except Exception as e:
        print(f"Error downloading data from DVC: {e}")
        return False
    
def only_dvc_in_dir(dvc_files_path: str):
    path = Path(dvc_files_path).resolve()
    list_dvc_files = list(path.glob('**/*.dvc'))
    list_all_files = [obj for obj in path.glob('**/*') if obj.is_file()]
    if len(list_all_files) == len(list_dvc_files):
        return True
    else:
        return False

if __name__ == "__main__":
    print(only_dvc_in_dir('../data'))