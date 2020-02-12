# -*- Coding: utf-8 -*-
# Packages
import os
import chainer
import math
import numpy as np
import chainer.functions as F
from chainer import Variable

# --- unused packages ---
# import glob
# import datetime
# import chainer.links as L
# from chainer import Chain, cuda, optimizers
# from PIL import Image

# MyPackages
import GlobalVariable as gv
import Experiment as ex

# kazuya
import pickle


# 学習するよ！
def Training():
    print("\r\nTraining...\r\n")
    for epoch in range(gv.G.EpochNum):
        # 学習関係
        totalLoss = 0
        totalAccuracy = 0
        for i in range(math.ceil((gv.G.DataSetSize - gv.G.Be.StartNowTrainNum) / gv.G.MinibatchSize)):
            # ミニバッチ学習
            x = Variable(gv.G.xp.array(gv.G.Be.GetFlattenData(gv.G.TrainDataIndex), dtype=gv.G.xp.float32))
            t = Variable(gv.G.xp.array(gv.G.Be.GetFlattenData(gv.G.TrainLabelIndex), dtype=gv.G.xp.int32))
            y = gv.G.Model(x)[gv.G.TargetLayer]
            gv.G.Model.zerograds()
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            loss.backward()
            gv.G.Optimizer.update()
            totalLoss += loss.data * gv.G.MinibatchSize
            totalAccuracy += acc.data * gv.G.MinibatchSize
            # 途中経過表示
            ydata = []
            for ngo in y.data[0:10]:
                ydata.append(int(np.argmax(chainer.cuda.to_cpu(ngo))))
            # End_For
            # print("t:{} y:{}".format(t[0:10], ydata) + " loss.data:{:*<8} acc.data:{:*<8}".format(
                # round(float(loss.data), 6), round(float(acc.data), 6)))
            print(f'\r{len(x)}')
            if gv.G.IsDebugLog:
                gv.G.LogFile.write("t:{} y:{} loss.data{:*<8} acc.data{:*<8}\r\n".format(
                    t[0:10], ydata, round(float(loss.data), 6), round(float(acc.data), 6)))
                gv.G.LogFile.flush()
            # End_If

            # 次のデータセットへｲｸｿﾞ!
            gv.G.Be.NextTrainData()
        # End_For(ミニバッチの終了)

        # エポック毎の誤差・Acc表示
        print(
            "---------------------------------------------↑{:^5}↑---------------------------------------------".format(epoch))
        print("TotalLoss:{}".format(totalLoss / gv.G.Be.TrainSize))
        print("TotalAcc:{}".format(totalAccuracy / gv.G.Be.TrainSize))
        print("---------------------------------------------------------------------------------------------------\r\n\r\n")

        # デバッグログ
        if gv.G.IsDebugLog:
            gv.G.LogFile.write(
                "---------------------------------------------↑{:^5}↑---------------------------------------------\r\n".format(epoch))
            gv.G.LogFile.write(
                "TotalLoss:{}\r\n".format(
                    totalLoss / gv.G.TrainSize))
            gv.G.LogFile.write(
                "TotalAcc:{}\r\n".format(
                    totalAccuracy /
                    gv.G.TrainSize))
            gv.G.LogFile.write(
                "---------------------------------------------------------------------------------------------------\r\n\r\n\r\n")
            gv.G.LogFile.flush()
        # End_If

        # 各エポック毎に認識実験する場合
        if gv.G.IsTestRecognitionEachEpoch:
            ex.UnknownDataExperiment(epoch)
            ex.KnownDataExperiment(epoch)
        # End_If

        # add kauzya
        if gv.G.is_every_save_pickle:
            print(f'  Save pickle: {epoch}')
            dir_path = f'{gv.G.SavePicklePath}/epoch/'

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            with open(f'{dir_path}{epoch}_epoch_{gv.G.DateTimeNow}.pkl', "wb") as p:
                pickle.dump(gv.G.Model, p)
        # ---
    # End_For(エポックの終了)
# End_Func
