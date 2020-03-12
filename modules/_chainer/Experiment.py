# -*- Coding: utf-8 -*-
# Packages
import chainer
import numpy as np
import chainer.functions as F
from chainer import Variable
from PIL import Image

# --- unused packages ---
# import os
# import os.path
# import glob
# import cv2
# import datetime
# import chainer.links as L
# from chainer import Chain, cuda, optimizers

# MyPackages
import Main_Utility as mu
import GlobalVariable as gv
from GradCAM import RunGradCAM as rgc

# import GetDatasetBebebe as gdb
# 未知画像認識実験


def UnknownDataExperiment(epoch):
    # 正答数
    accCounts = [0] * len(gv.G.Be.DirNames)
    # 認識実験用データ・正解ラベル
    testData = gv.G.Be.GetFlattenData(gv.G.TestDataIndex)
    testLabel = gv.G.Be.GetFlattenData(gv.G.TestLabelIndex)

    # デバッグ表示
    print("Testing Unknown Data...")
    print("Testdata Shape", testData.shape)
    if gv.G.IsDebugLog:
        gv.G.LogFile.write("Testing Unknown Data...\r\n")
        gv.G.LogFile.write("Testdata Shape : {}\r\n".format(testData.shape))
        gv.G.LogFile.flush()
    # End_If

    # 実験ｲｸｿﾞ!
    for i in range(len(testData)):
        # 進捗
        print("\r{} / {}".format(i, len(testData)), end="")

        # 1毎ずつ実験
        x = Variable(gv.G.xp.array([testData[i]], dtype=gv.G.xp.float32))
        t = Variable(gv.G.xp.array([testLabel[i]], dtype=gv.G.xp.int32))
        ydict = gv.G.Model(x)
        y = ydict[gv.G.TargetLayer]
        y = F.softmax(y.data)

        # 正解・不正解を判定，不正解ならその画像を保存
        ans = int(gv.G.xp.argmax(y.data))  # 回答
        right = int(t.data[0])  # 正解
        # 正解ならカウント
        if ans == right:
            accCounts[right] += 1
            # GradCAMする場合
            if gv.G.IsGradCAM:
                pilImg = None
                if gv.G.IsGPU:
                    pilImg = Image.fromarray(
                        np.uint8(
                            chainer.cuda.to_cpu(
                                testData[i]).transpose(
                                1, 2, 0)))
                else:
                    pilImg = Image.fromarray(
                        np.uint8(
                            testData[i]).transpose(
                            1, 2, 0))
                # End_IfElse
                # 保存
                ggcam, gbp, gcam = rgc.RunGradCAM(pilImg, right)
                dirName = "Epoch_" + \
                    str(epoch) if epoch >= 0 else "Training_Finished_Experiment"
                ggcam = mu.OpenCVtoPIL(ggcam)
                gbp = mu.OpenCVtoPIL(gbp)
                gcam = mu.OpenCVtoPIL(gcam)
                ggcam.save(gv.G.GradCAM_ImagePath + "/Correct/Unknown/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_ggcam_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
                gbp.save(gv.G.GradCAM_ImagePath + "/Correct/Unknown/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_gbp_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
                gcam.save(gv.G.GradCAM_ImagePath + "/Correct/Unknown/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_gcam_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
            # End_If
        # 不正解なら誤認識画像として保存
        else:
            pilImg = None
            if gv.G.IsGPU:
                pilImg = Image.fromarray(
                    np.uint8(
                        chainer.cuda.to_cpu(
                            testData[i]).transpose(
                            1, 2, 0)))
            else:
                pilImg = Image.fromarray(
                    np.uint8(
                        testData[i]).transpose(
                        1, 2, 0))
            # End_IfElse

            # GradCAMする場合
            if gv.G.IsGradCAM:
                ggcam, gbp, gcam = rgc.RunGradCAM(pilImg, right)
                # 保存
                dirName = "Epoch_" + \
                    str(epoch) if epoch >= 0 else "Training_Finished_Experiment"
                ggcam = mu.OpenCVtoPIL(ggcam)
                gbp = mu.OpenCVtoPIL(gbp)
                gcam = mu.OpenCVtoPIL(gcam)
                ggcam.save(gv.G.GradCAM_ImagePath + "/False/Unknown/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_ggcam_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
                gbp.save(gv.G.GradCAM_ImagePath + "/False/Unknown/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_gbp_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
                gcam.save(gv.G.GradCAM_ImagePath + "/False/Unknown/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_gcam_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
            # End_If

            # エポック・クラスを分けて保存
            dirName = "Epoch_" + \
                str(epoch) if epoch >= 0 else "Training_Finished_Experiment"
            pilImg = pilImg.resize((500, 500))
            pilImg.save(gv.G.FalsePath + "/Unknown/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                "\\")[-1] + "/ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
        # End_IfElse
    # End_For(1枚の画像)

    # 実験結果表示
    print("Test_Num : {}".format(len(testData)))
    for i in range(len(accCounts)):
        print("{}  acc : {}  rate : {}".format(gv.G.Be.DirNames[i].split(
            "\\")[-1], accCounts[i], accCounts[i] / gv.G.TestSize))
    # End_For
    print("Total acc :{}".format(sum(accCounts) / len(testData)))
    print()

    # デバッグ表示
    if gv.G.IsDebugLog:
        gv.G.LogFile.write("Test Num : {}\r\n".format(len(testData)))
        for i in range(len(accCounts)):
            gv.G.LogFile.write("{}  acc : {}  rate : {}\r\n".format(
                gv.G.Be.DirNames[i].split("\\")[-1], accCounts[i], accCounts[i] / gv.G.TestSize))
        # End_For
        gv.G.LogFile.write(
            "Total acc : {}\r\n\r\n".format(
                sum(accCounts) /
                len(testData)))
        gv.G.LogFile.flush()
    # End_If

    # エポック毎のレート出力
    if gv.G.IsRateLog:
        gv.G.RateLogFile.write(
            "{},,".format(
                epoch if epoch >= 0 else "Training Finished"))
        for i in range(len(accCounts)):
            gv.G.RateLogFile.write("{},".format(accCounts[i] / gv.G.TestSize))
        # End_For
        gv.G.RateLogFile.flush()
    # End_If
# End_Func

# 既知画像認識実験


def KnownDataExperiment(epoch):
    # 正答データ
    accCounts = [0] * len(gv.G.Be.DirNames)
    # 認識実験用データ・正解ラベル
    testData = gv.G.Be.GetFlattenData(gv.G.TrainedTestDataIndex)
    testLabel = gv.G.Be.GetFlattenData(gv.G.TrainedTestLabelIndex)

    # デバッグ表示
    print("Testing known Data...")
    print("Testdata Shape", testData.shape)
    if gv.G.IsDebugLog:
        gv.G.LogFile.write("Testing known Data...\r\n")
        gv.G.LogFile.write("Testdata Shape : {}\r\n".format(testData.shape))
        gv.G.LogFile.flush()
    # End_If

    # 実験ｲｸｿﾞ!
    for i in range(len(testData)):
        # 進捗
        print("\r{} / {}".format(i, len(testData)), end="")

        # 1毎ずつ実験
        x = Variable(gv.G.xp.array([testData[i]], dtype=gv.G.xp.float32))
        t = Variable(gv.G.xp.array([testLabel[i]], dtype=gv.G.xp.int32))
        ydict = gv.G.Model(x)
        y = ydict[gv.G.TargetLayer]
        y = F.softmax(y.data)

        # 正解・不正解を判定，不正解ならその画像を保存
        ans = int(gv.G.xp.argmax(y.data))  # 回答
        right = int(t.data[0])  # 正解
        # 正解ならカウント
        if ans == right:
            accCounts[right] += 1

            # GradCAMする場合
            if gv.G.IsGradCAM:
                pilImg = None
                if gv.G.IsGPU:
                    pilImg = Image.fromarray(
                        np.uint8(
                            chainer.cuda.to_cpu(
                                testData[i]).transpose(
                                1, 2, 0)))
                else:
                    pilImg = Image.fromarray(
                        np.uint8(
                            testData[i]).transpose(
                            1, 2, 0))
                # End_IfElse
                ggcam, gbp, gcam = rgc.RunGradCAM(pilImg, right)
                # 保存
                dirName = "Epoch_" + \
                    str(epoch) if epoch >= 0 else "Training_Finished_Experiment"
                ggcam = mu.OpenCVtoPIL(ggcam)
                gbp = mu.OpenCVtoPIL(gbp)
                gcam = mu.OpenCVtoPIL(gcam)
                ggcam.save(gv.G.GradCAM_ImagePath + "/Correct/Known/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_ggcam_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
                gbp.save(gv.G.GradCAM_ImagePath + "/Correct/Known/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_gbp_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
                gcam.save(gv.G.GradCAM_ImagePath + "/Correct/Known/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_gcam_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
            # End_If
        # 不正解なら誤認識画像として保存
        else:
            pilImg = None
            if gv.G.IsGPU:
                pilImg = Image.fromarray(
                    np.uint8(
                        chainer.cuda.to_cpu(
                            testData[i]).transpose(
                            1, 2, 0)))
            else:
                pilImg = Image.fromarray(
                    np.uint8(
                        testData[i]).transpose(
                        1, 2, 0))
            # End_IfElse

            # GradCAMする場合
            if gv.G.IsGradCAM:
                ggcam, gbp, gcam = rgc.RunGradCAM(pilImg, right)
                # 保存
                dirName = "Epoch_" + \
                    str(epoch) if epoch >= 0 else "Training_Finished_Experiment"
                ggcam = mu.OpenCVtoPIL(ggcam)
                gbp = mu.OpenCVtoPIL(gbp)
                gcam = mu.OpenCVtoPIL(gcam)
                ggcam.save(gv.G.GradCAM_ImagePath + "/False/Known/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_ggcam_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
                gbp.save(gv.G.GradCAM_ImagePath + "/False/Known/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_gbp_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
                gcam.save(gv.G.GradCAM_ImagePath + "/False/Known/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                    "\\")[-1] + "/" + str(i) + "_gcam_ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
            # End_If

            # エポック・クラスを分けて保存
            dirName = "Epoch_" + \
                str(epoch) if epoch >= 0 else "Training_Finished_Experiment"
            pilImg = pilImg.resize((500, 500))
            pilImg.save(gv.G.FalsePath + "/Known/" + dirName + "/" + gv.G.Be.DirNames[right].split(
                "\\")[-1] + "/ans[" + str(ans) + "]_label[" + str(right) + "]_" + str(i) + ".png")
        # End_Else
    # End_For(1毎の画像)

    # 実験結果表示
    print("Test_Num : {}".format(len(testData)))
    for i in range(len(accCounts)):
        print("{}  acc : {}  rate : {}".format(gv.G.Be.DirNames[i].split(
            "\\")[-1], accCounts[i], accCounts[i] / gv.G.TestSize))
    # End_For
    print("Total acc :{}".format(sum(accCounts) / len(testData)))
    print("\r\n\r\n")

    # デバッグ表示
    if gv.G.IsDebugLog:
        gv.G.LogFile.write("Test Num : {}\r\n".format(len(testData)))
        for i in range(len(accCounts)):
            gv.G.LogFile.write("{}  acc : {}  rate : {}\r\n".format(
                gv.G.Be.DirNames[i].split("\\")[-1], accCounts[i], accCounts[i] / gv.G.TestSize))
        # End_For
        gv.G.LogFile.write(
            "Total acc : {}\r\n\r\n\r\n".format(
                sum(accCounts) / gv.G.TestSize))
        gv.G.LogFile.flush()
    # End_If

    # エポック毎のレート出力
    if gv.G.IsRateLog:
        #gv.G.RateLogFile.write("{},,".format(epoch if epoch >= 0 else "Training Finished"))
        for i in range(len(accCounts)):
            gv.G.RateLogFile.write("{},".format(accCounts[i] / gv.G.TestSize))
        # End_For
        gv.G.RateLogFile.write("\r\n")
        gv.G.RateLogFile.flush()
    # End_If
# End_Func
