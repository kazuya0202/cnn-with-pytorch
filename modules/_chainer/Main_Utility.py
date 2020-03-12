# -*- Coding: utf-8 -*-
# Packages
import os
import os.path
import glob
import cv2
import pickle
import numpy as np
from PIL import Image
from chainer import cuda, optimizers

# --- unused packages ---
# import chainer
# import datetime
# import chainer.links as L
# import chainer.functions as F
# from chainer import Chain, Variable

# MyPackages
import GlobalVariable as gv
import Train as tr
import Experiment as ex
import CNN as cnn
import GetDatasetBebebe as gdb

# メイン文


def Main():
    # 初期化
    Initialize()
    # 学習済みモデルを使用する場合
    if gv.G.IsLoadPickle:
        with open(gv.G.LoadPicklePath, "rb") as p:
            gv.G.Model = pickle.load(p)
        # End_With
    # 学習する場合
    else:
        tr.Training()
    # End_IfElse
    # 実験
    ex.UnknownDataExperiment(-1)
    ex.KnownDataExperiment(-1)
    # 保存
    SaveModelAsPickle()
# End_Func

# いろいろ初期化


def Initialize():
    # ペンギンOS
    if gv.G.IsLinux:
        gv.G.ImagePath = gv.G.ImagePath.replace("\\", "/")
    # End_If

    # テスト画像を上書きするときはすべての画像が学習用画像になる
    if gv.G.IsOverwriteTestData:
        gv.G.TrainSize += gv.G.TestSize
    # End_IF

    # モデル生成
    gv.G.Model = cnn.CNN()
    gv.G.Optimizer = optimizers.Adam()
    gv.G.Optimizer.setup(gv.G.Model)

    # GPUを使う場合はモデルをGPUに
    gv.G.xp = cuda.cupy if gv.G.IsGPU else np
    if gv.G.IsGPU:
        gpuDevice = 0
        cuda.get_device(gpuDevice).use()
        gv.G.Model.to_gpu(gpuDevice)
    # End_If

    # データセット取得オブジェクト生成，1回目のデータセット取得
    print("Make Instance and Get Dataset...")
    gv.G.Be = gdb.GetDatasetBebebe(
        gv.G.ImagePath,
        gv.G.Extention,
        gv.G.DataSetSize,
        gv.G.TestSize,
        gv.G.MinibatchSize,
        gv.G.Width,
        gv.G.Height,
        gv.G.IsOverwriteTestData,
        gv.G.OverwriteTestDataPath,
        gv.G.IsOverwriteTrainedTestData,
        gv.G.OverwriteTrainedTestDataPath)

    # DebugLog
    if gv.G.IsDebugLog:
        txtFiles = glob.glob(gv.G.LogFilePath + "/" + "*.txt")
        gv.G.LogFile = open(gv.G.LogFilePath + "/" + str(len(txtFiles)) + "_" + gv.G.DateTimeNow + ".txt", "w")
        gv.G.LogFile.write("DatasetSize : {}  TrainDataSize : {}  TestDataSize : {}  EpochNum : {}\r\n".format(
                gv.G.DataSetSize,
                gv.G.TrainSize,
                gv.G.TestSize,
                gv.G.EpochNum))
        gv.G.LogFile.write("ImagePath :{}\r\n".format(gv.G.ImagePath))
        gv.G.LogFile.write("OverwriteTestDataPath : {}\r\n".format(
            gv.G.OverwriteTestDataPath if gv.G.IsOverwriteTestData else "None"))
        gv.G.LogFile.write("OverwriteTrainedTestDataPath : {}\r\n".format(
            gv.G.OverwriteTrainedTestDataPath if gv.G.IsOverwriteTrainedTestData else "None"))
        gv.G.LogFile.flush()
    # End_If

    # RateLog
    if gv.G.IsRateLog:
        dirNames = glob.glob("{}/*".format(gv.G.ImagePath))
        txtFiles = glob.glob(gv.G.RateLogFilePath + "/" + "*.csv")
        gv.G.RateLogFile = open(gv.G.RateLogFilePath +
                                "/" +
                                str(len(txtFiles)) +
                                "_" +
                                gv.G.DateTimeNow +
                                ".csv", "w")
        # 見出し
        gv.G.RateLogFile.write("データNo,Unknown,")
        for d in dirNames:
            gv.G.RateLogFile.write(d.split("\\")[-1] + ",")
        gv.G.RateLogFile.write("Known")
        for d in dirNames:
            gv.G.RateLogFile.write("," + d.split("\\")[-1])
        gv.G.RateLogFile.write("\r\n")
        gv.G.RateLogFile.flush()
    # End_IF

    # 誤認識画像パス作成
    MakeFalseDir()

    # デバッグ表示とか
    # ディレクトリネームについて
    for i in range(len(gv.G.Be.DirNames)):
        print("DirNames_{}  {}".format(i, gv.G.Be.DirNames[i].split("\\")[-1]))
        if gv.G.IsDebugLog:
            gv.G.LogFile.write("DirNames_{}  {}\r\n".format(
                i, gv.G.Be.DirNames[i].split("\\")[-1]))
            gv.G.LogFile.flush()
        # End_For
    # End_For
# End_Func

# FalsePath直下に新規フォルダを作成して誤認識画像をフォルダに保存


def MakeFalseDir():
    print("Make False Images Directorys...")
    # FalsePathまでを再帰的に作成
    os.makedirs(gv.G.FalsePath, exist_ok=True)
    # FalsePath直下に今回の誤認識画像を保存するフォルダを作成する
    dirs = [p for p in glob.glob(gv.G.FalsePath + "/*") if os.path.isdir(p)]
    gv.G.FalsePath = gv.G.FalsePath + "/No" + \
        str(len(dirs)) + "_" + gv.G.DateTimeNow
    os.makedirs(gv.G.FalsePath)
    os.makedirs(gv.G.FalsePath + "/Known")
    os.makedirs(gv.G.FalsePath + "/Unknown")

    # 各エポック毎の誤認識画像パス
    for epoch in range(gv.G.EpochNum):
        os.makedirs(gv.G.FalsePath + "/Known/Epoch_" + str(epoch))
        os.makedirs(gv.G.FalsePath + "/Unknown/Epoch_" + str(epoch))
        # 各クラスごとの誤認識画像パス
        for d in gv.G.Be.DirNames:
            os.makedirs(gv.G.FalsePath + "/Known/Epoch_" +
                        str(epoch) + "/" + d.split("\\")[-1])
            os.makedirs(gv.G.FalsePath + "/Unknown/Epoch_" +
                        str(epoch) + "/" + d.split("\\")[-1])
        # End_For
    # End_For

    # 学習終了後の認識実験(エポック毎に認識する際は最後のEpochと同じ内容(のはず))
    os.makedirs(gv.G.FalsePath + "/Known/Training_Finished_Experiment")
    os.makedirs(gv.G.FalsePath + "/Unknown/Training_Finished_Experiment")
    for d in gv.G.Be.DirNames:
        os.makedirs(gv.G.FalsePath +
                    "/Known/Training_Finished_Experiment/" + d.split("\\")[-1])
        os.makedirs(
            gv.G.FalsePath + "/Unknown/Training_Finished_Experiment/" + d.split("\\")[-1])
    # End_For

    # Grad-CAMを使う場合
    if gv.G.IsGradCAM:
        os.makedirs(gv.G.GradCAM_ImagePath, exist_ok=True)
        dirs = [
            p for p in glob.glob(
                gv.G.GradCAM_ImagePath + "/*") if os.path.isdir(p)]
        gv.G.GradCAM_ImagePath = gv.G.GradCAM_ImagePath + \
            "/No" + str(len(dirs)) + "_" + gv.G.DateTimeNow
        os.makedirs(gv.G.GradCAM_ImagePath)
        for s in ["/Correct", "/False"]:
            os.makedirs(gv.G.GradCAM_ImagePath + s + "/Known")
            os.makedirs(gv.G.GradCAM_ImagePath + s + "/Unknown")
            # 各エポック毎の誤認識画像パス
            for epoch in range(gv.G.EpochNum):
                os.makedirs(gv.G.GradCAM_ImagePath + s + "/Known/Epoch_" + str(epoch))
                os.makedirs(gv.G.GradCAM_ImagePath + s + "/Unknown/Epoch_" + str(epoch))
                # 各クラスごとの誤認識画像パス
                for d in gv.G.Be.DirNames:
                    os.makedirs(gv.G.GradCAM_ImagePath + s +
                                "/Known/Epoch_" + str(epoch) + "/" + d.split("\\")[-1])
                    os.makedirs(gv.G.GradCAM_ImagePath + s +
                                "/Unknown/Epoch_" + str(epoch) + "/" + d.split("\\")[-1])
                # End_For
            # End_For
            # 学習終了後の認識実験(エポック毎に認識する際は最後のEpochと同じ内容(のはず))
            os.makedirs(
                gv.G.GradCAM_ImagePath +
                s +
                "/Known/Training_Finished_Experiment")
            os.makedirs(
                gv.G.GradCAM_ImagePath +
                s +
                "/Unknown/Training_Finished_Experiment")
            for d in gv.G.Be.DirNames:
                os.makedirs(gv.G.GradCAM_ImagePath + s +
                            "/Known/Training_Finished_Experiment/" + d.split("\\")[-1])
                os.makedirs(gv.G.GradCAM_ImagePath + s +
                            "/Unknown/Training_Finished_Experiment/" + d.split("\\")[-1])
            # End_For
        # End_For
    # End_If
# End_Func

# OpenCV画像をPIL画像に変換


def OpenCVtoPIL(cvImg):
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    cvImg = np.array(cvImg, dtype=np.uint8)
    pilImg = Image.fromarray(cvImg)
    pilImg = pilImg.convert("RGB")
    return pilImg
# End_Func

# PIL画像をOpenCV画像に変換


def PILtoOPenCV(pilImg):
    cvImg = np.array(pilImg)
    if cvImg.ndim == 2:  # モノクロ
        pass
    elif cvImg.ndim == 3:  # カラー
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)
    elif cvImg.ndim == 4:  # 透過
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGBA2BGRA)
    # End_IfElse
    return cvImg
# End_Func

# GlobalVariableのモデルをPickleとして保存する


def SaveModelAsPickle():
    pickles = glob.glob(gv.G.SavePicklePath + "/" + "*.pickle")
    # 保存
    if not gv.G.IsLoadPickle:
        print("Pickle開始")
        with open(gv.G.SavePicklePath + "/" + str(len(pickles)) + "_" + gv.G.DateTimeNow + ".pickle", "wb") as p:
            pickle.dump(gv.G.Model, p)
        # End_With
    # End_If
# End_Func


if __name__ == "__main__":
    Main()
