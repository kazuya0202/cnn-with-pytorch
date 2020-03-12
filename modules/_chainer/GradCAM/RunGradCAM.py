# -*- Coding: utf-8 -*-
# Packages
import cv2
import chainer
import copy
import numpy as np
# MyPackages
import GlobalVariable as gv
import Main_Utility as mu
from GradCAM import BackProp as bp

# 画像を投げるとGradCAM画像を生成する


def RunGradCAM(img, label):
    gradCAM = bp.GradCAM(gv.G.Model)
    guidedBackprop = bp.GuidedBackprop(gv.G.Model)

    src = mu.PILtoOPenCV(img)
    src = cv2.resize(src, (gv.G.Width, gv.G.Height))
    x = src.astype(np.float32) - np.float32([103.939, 116.779, 123.68])
    x = x.transpose(2, 0, 1)[np.newaxis, :, :, :]

    gcam = gradCAM.generate(x, label, gv.G.GradCAM_Layer)
    gcam = np.uint8(gcam * 255 / gcam.max())
    gcam = cv2.resize(gcam, (gv.G.Width, gv.G.Height))
    gbp = guidedBackprop.generate(x, label, gv.G.TargetLayer)

    ggcam = gbp * gcam[:, :, np.newaxis]
    ggcam -= ggcam.min()
    ggcam = 255 * ggcam / ggcam.max()

    gbp -= gbp.min()
    gbp = 255 * gbp / gbp.max()

    heatmap = cv2.applyColorMap(gcam, cv2.COLORMAP_JET)
    gcam = np.float32(src) + np.float32(heatmap)
    gcam = 255 * gcam / gcam.max()
    return ggcam, gbp, gcam
# End_Func
