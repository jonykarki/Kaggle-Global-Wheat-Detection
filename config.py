
import os
import subprocess
import kaggle
from easydict import EasyDict as edict

_C = edict()
config = _C

_C.BASEPATH = "/content"

# Data folders
_C.DATA = edict()
_C.DATA.BASE = os.path.join(_C.BASEPATH, "data")
_C.DATA.COMP_NAME = "global-wheat-detection"

def download_kaggle_data():
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(_C.DATA.COMP_NAME, path=_C.DATA.BASE, quiet=False)
    subprocess.call("unzip \*.zip -q", shell=True, cwd=_C.DATA.BASE)
    subprocess.call("rm *.zip", shell=True, cwd=_C.DATA.BASE)

if __name__ == "__main__":
    download_kaggle_data()