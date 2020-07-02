"""
import sys, os
if os.path.abspath(os.pardir) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.pardir))
import CONFIG

%reload_ext autoreload
%autoreload 2
"""

import os
import json
import subprocess
import kaggle
from easydict import EasyDict as edict

_C = edict()
CFG = _C

_C.BASEPATH = "/content"
_C.THISPATH = os.path.dirname(os.path.abspath(__file__))
_C.MODEL_OUTPUT_PATH = "out"

# Data folders
_C.DATA = edict()
_C.DATA.BASE = os.path.join(_C.BASEPATH, "data")
_C.DATA.COMP_NAME = "global-wheat-detection"
_C.DATA.MODELS_OUT = os.path.join(_C.THISPATH, _C.MODEL_OUTPUT_PATH)


def download_kaggle_data():
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(
        _C.DATA.COMP_NAME, path=_C.DATA.BASE, quiet=False)
    subprocess.call("unzip -q \*.zip", shell=True, cwd=_C.DATA.BASE)
    subprocess.call("rm *.zip", shell=True, cwd=_C.DATA.BASE)


def upload_to_kaggle(slug, title, new=False, msg="new version"):
    meta = {
        "licenses": [
            {
                "name": "CC0-1.0"
            }
        ]
    }
    meta["id"] = f"jonykarki/{slug}"
    meta["title"] = title
    with open(os.path.join(_C.DATA.MODELS_OUT,"dataset-metadata.json"), "w") as out:
        json.dump(meta, out)
    if new == True:
        subprocess.call(f"kaggle datasets create -p {_C.MODEL_OUTPUT_PATH}", shell=True, cwd=_C.THISPATH)
    else:
        subprocess.call(f"kaggle datasets version -p {_C.MODEL_OUTPUT_PATH} -m '{msg}'", shell=True, cwd=_C.THISPATH)

def init():
    if not os.path.exists(_C.MODEL_OUTPUT_PATH):
        os.makedirs(_C.MODEL_OUTPUT_PATH)
    download_kaggle_data()
    subprocess.call(f"python3 -m pip install --upgrade -r requirements.txt", shell=True)

if __name__ == "__main__":
    init()
