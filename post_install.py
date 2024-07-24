import subprocess
import sys

def install_mmcv():
    try:
        subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmcv>=2.2.0'])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install mmcv: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_mmcv()
