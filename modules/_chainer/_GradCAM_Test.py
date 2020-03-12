# -*- Coding: utf-8 -*-
#Packages
import glob
import pickle
from PIL import Image
from chainer import cuda
import cv2
import numpy as np

# --- unused packages ---
# import datetime
# import os
# import os.path
# import chainer
# from chainer import Chain, Variable, optimizers
# import chainer.functions as F
# import chainer.links as L

import GlobalVariable as gv
from GradCAM import RunGradCAM as rgc
# import CNN as cnn
# import Experiment as ex
# import GetDatasetBebebe as gdb
# import Main_Utility as mu
# import Train as tr


#メイン文
def Main():
	gv.G.xp = cuda.cupy
	#学習済みモデルを使用する場合
	with open(r"C:\Okabe\D1\20190711_3クラス実験\Model.pickle", "rb") as p:
		gv.G.Model = pickle.load(p)

	unti = r"C:\Okabe\D1\論文用FalseImageGradCAM実験\3class"
	imgs = glob.glob(unti + "\\*.bmp")
	print(imgs)
	for i in range(int(len(imgs)/2)):
		ggcam, gbp, gcam = rgc.RunGradCAM(Image.fromarray(np.array(Image.open(imgs[i]), dtype=np.uint8)), 2)
		cv2.imwrite(unti + "\\" + str(i) + "_ggcam.bmp", ggcam)
		cv2.imwrite(unti + "\\" + str(i) + "_gcam.bmp", gcam)
		cv2.imwrite(unti + "\\" + str(i) + "_gbp.bmp", gbp)
		print("unti")

	for i in range(int(len(imgs)/2) ):
		ggcam, gbp, gcam = rgc.RunGradCAM(Image.fromarray(np.array(Image.open(imgs[2+i]), dtype=np.uint8)), 1)
		cv2.imwrite(unti + "\\" + str(i+2) + "_ggcam.bmp", ggcam)
		cv2.imwrite(unti + "\\" + str(i+2) + "_gcam.bmp", gcam)
		cv2.imwrite(unti + "\\" + str(i+2) + "_gbp.bmp", gbp)
		print(unti + "\\" + str(i+2) + "_ggcam.bmp")
#End_Func

if __name__ == "__main__":
	Main()
