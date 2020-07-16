import sys, os
sys.path.insert(0, "/content/Kaggle-Global-Wheat-Detection")
import CONFIG

sys.path.insert(0, os.path.join(CONFIG.CFG.BASEPATH, "Yet-Another-EfficientDet-Pytorch"))

import torch
import numpy as np 
import pandas as pd
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, postprocess

force_input_size = None
compound_coef = 4

anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

model = EfficientDetBackbone(compound_coef=4, num_classes=1, ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load("efficientdet-d4_14_20235.pth"))
model.requires_grad_(False)
model.eval()

IMG_PATH = [os.path.join(CONFIG.CFG.DATA.BASE, "test", img_path) for img_path in os.listdir(os.path.join(CONFIG.CFG.DATA.BASE, "test"))]

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

ori_imgs, framed_imgs, framed_metas = preprocess(*IMG_PATH, max_size=input_size)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(DEVICE)

x = torch.stack([torch.from_numpy(fi).to(DEVICE) for fi in framed_imgs], 0)

x.to(torch.float32).permute(0, 3, 1, 2)

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    ClipBoxes = ClipBoxes()

    out = postprocess(x, anchors, regression, classification, regressBoxes, ClipBoxes, 0.2, 0.2)

    print(out)