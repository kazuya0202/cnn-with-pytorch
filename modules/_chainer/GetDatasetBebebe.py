# -*- Coding: utf-8 -*-
# Packages
import glob
import numpy as np
from PIL import Image

# --- unused packages ---
# import os
# import math
# import copy
# import chainer
# from chainer import Chain, Variable, cuda

# MyPackages
import GlobalVariable as gv

#-------------------------------------------------------------------------------------------#
##分割してデータを取得できるゾ																	#
##データはNumpy配列なのでGPU使いたいときはTrainer側で頑張って									#
#																							#
##使い方																						#
##1.  インスタンス化すると自動でテストデータと1回目のトレインデータを取得							#
##2.  GetFlattenを使ってデータを取得															#
##3.  NextTrainDataを使って次のデータ読みこみ													#
##4.  2~3を繰り返す																			#
#-------------------------------------------------------------------------------------------#


class GetDatasetBebebe:
    # Constructor
    def __init__(
            self,
            imgDir,
            extentions,
            datasetSize,
            testSize,
            minibatchSize,
            width=60,
            height=60,
            isOverwriteTestData=False,
            overwriteTestDataPath=None,
            isOverwriteTrainedTestData=False,
            overwriteTrainedTestDataPath=None):
        # GlobalVariable依存の変数
        self.ImgDir = imgDir  # 画像読み込み元パス
        self.Extentions = extentions  # 使用する拡張子
        self.DatasetSize = datasetSize  # 1クラスでの学習画像+テスト画像
        self.TestSize = testSize  # テスト用画像
        self.TrainSize = self.DatasetSize - self.TestSize  # 学習用画像
        self.MinibatchSize = minibatchSize  # ミニバッチサイズ
        self.Width = width  # 使用する画像の横幅(リサイズされる)
        self.Height = height  # 使用する画像の縦幅(リサイズされる)
        self.IsOverwriteTestData = isOverwriteTestData  # テストデータを任意の画像で上書きするか
        self.OverwriteTestDataPath = overwriteTestDataPath  # テストデータを上書きする画像ディレクトリ
        # 既知画像認識実験用画像を任意のデータで上書きするか
        self.IsOverwriteTrainedTestData = isOverwriteTrainedTestData
        # 既知画像認識実験用画像を上書きする画像を格納した画像ディレクトリ
        self.OverwriteTrainedTestDataPath = overwriteTrainedTestDataPath

        # 独自の変数
        self.DirNames = glob.glob("{}/*".format(self.ImgDir))  # 各クラスのディレクトリ名
        self.FileNames = self.__GetAllImagePath(
            self.ImgDir).copy()  # 各クラスのすべての画像パス
        self.NowTrainNum = 0  # 学習用画像の現在学習した位置(イテレータ的)
        # 学習用画像開始のインデックス(テスト用画像を上書きするときは0になる)
        self.StartNowTrainNum = self.TestSize

        # 並び替え用変数
        # データセット全体を並び替え(今後一切弄らない)
        self.Perms = [np.random.permutation(
            len(self.FileNames[i])) for i in range(len(self.DirNames))]
        # ミニバッチサイズを取得するなら
        self.FlattenTrainPerm = np.array(np.random.permutation(
            self.MinibatchSize * len(self.DirNames)))
        self.FlattenTestPerm = np.array(
            np.random.permutation(self.TestSize * len(self.DirNames)))
        # 既知画像の認識実験用(毎回同じ既知画像が出てくると嬉しい)
        self.FlattenTrainedTestPerm = np.array(
            np.random.permutation(self.TestSize * len(self.DirNames)))

        # データセット格納用
        self.TrainData = []  # 学習用画像
        self.TrainLabel = []  # 学習用画像正解ラベル
        self.TestData = []  # テスト用画像
        self.TestLabel = []  # テスト用画像正解ラベル
        self.TrainedTestData = []  # 既知画像の認識実験用画像
        self.TrainedTestLabel = []  # 既知画像の認識実験用画像正解ラベル

        # いろいろ初期化
        self.InitializeImages()

    # End_Constructor

    # 1回目のテスト・学習データ取得，既知画像認識実験画像取得，テスト・既知画像認識実験画像書き換え
    def InitializeImages(self):
        # テスト用画像取得
        print("At First TestData Getting...")
        self.TestData, self.TestLabel = self.__GetDatasetBebebe(
            0, self.TestSize)
        self.NowTrainNum += self.TestSize

        # 既知画像実験用画像取得
        print("\r\nTrainedTestData Getting...")
        self.TrainedTestData, self.TrainedTestLabel = self.__GetDatasetBebebe(
            self.NowTrainNum, self.NowTrainNum + self.TestSize)

        # テストデータを入れ替える場合は学習用画像は0からすべての画像になる
        if self.IsOverwriteTestData:
            # 0から最後まですべて学習用画像
            self.NowTrainNum = 0
            self.StartNowTrainNum = 0
            self.TrainSize += self.TestSize
            # テストデータ入れ替え
            self.__ChangeTestDataImage()
        # End_If

        # 既知画像認識実験用の画像を入れ替える
        if self.IsOverwriteTrainedTestData:
            self.__ChangeTrainedTestDataImage()
        # End_If

        # 学習用画像取得(ミニバッチ分だけ)
        print("\r\nAt First TrainData Getting...")
        self.TrainData, self.TrainLabel = self.__GetDatasetBebebe(
            self.NowTrainNum, self.NowTrainNum + self.MinibatchSize)
        self.NowTrainNum += self.MinibatchSize
    # End_Method

    # 任意の範囲のデータセット取得

    def __GetDatasetBebebe(self, startIndex, endIndex):
        # Initialize
        imageData = [[] for d in self.DirNames]
        label = []
        # 各クラスごとに画像を取得していく
        for i in range(len(self.DirNames)):
            for j in range(startIndex, min(endIndex, len(self.FileNames[i]))):
                img = Image.open(self.FileNames[i][self.Perms[i][j]])
                img = img.resize((self.Width, self.Height))

                # kazuya
                # remove alpha channel
                img = img.convert('RGB')

                img = np.array(img, dtype=np.float32)
                img = img[:, :, :3]
                img = img.transpose(2, 0, 1)
                imageData[i].append(img)
            # End_For
        # End_For

        # ラベル
        for i in range(len(imageData)):
            label.append([i] * len(imageData[i]))
        # End_For

        # ランダムにする必要性なし → GetFlatten時にランダム
        imageData = np.array(imageData, dtype=np.float32)
        label = np.array(label, dtype=np.int32)

        return imageData, label
    # End_Method

    def NextTrainData(self):
        # 次の範囲
        max = min(self.DatasetSize, self.NowTrainNum + self.MinibatchSize)

        # 学習用画像枚数とミニバッチ枚数が同じ時の例外処理
        if self.MinibatchSize == self.TrainSize:
            self.NowTrainNum = self.StartNowTrainNum if max == self.DatasetSize else self.NowTrainNum + self.MinibatchSize
        # End_If

        # 次の学習用画像取得
        self.TrainData, self.TrainLabel = self.__GetDatasetBebebe(
            self.NowTrainNum, max)

        # 並び替え乱数の更新
        self.FlattenTrainPerm = np.array(np.random.permutation(min(
            self.MinibatchSize * len(self.DirNames), (self.DatasetSize - self.NowTrainNum) * len(self.DirNames))))

        # 現在の取得位置(イテレータの更新)
        self.NowTrainNum = self.StartNowTrainNum if max == self.DatasetSize else self.NowTrainNum + self.MinibatchSize

        # 学習用画像枚数とミニバッチ枚数が同じ時の例外処理
    # End_Method

    # クラスごとに配列にされているミニバッチを1次元配列にして返す
    def GetFlattenData(self, kind):
        hotti = None
        # 学習画像が欲しい場合
        if kind == gv.G.TrainDataIndex:
            hotti = self.TrainData[0].copy().tolist()
            for i in range(1, len(self.DirNames)):
                hotti.extend(self.TrainData[i].tolist())
            # End_For
            hotti = np.array(hotti, dtype=np.float32)
            return hotti[self.FlattenTrainPerm]

        # 学習画像のラベルが欲しい場合
        elif kind == gv.G.TrainLabelIndex:
            hotti = self.TrainLabel[0].copy().tolist()
            for i in range(1, len(self.TrainLabel)):
                hotti.extend(self.TrainLabel[i].tolist())
            # End_For
            hotti = np.array(hotti, dtype=np.int32)
            return hotti[self.FlattenTrainPerm]
        # テスト画像が欲しい場合
        elif kind == gv.G.TestDataIndex:
            hotti = self.TestData[0].copy().tolist()
            for i in range(1, len(self.TestData)):
                hotti.extend(self.TestData[i].tolist())
            # End_For
            hotti = np.array(hotti, dtype=np.float32)
            return hotti[self.FlattenTestPerm]
        # テスト画像のラベルが欲しい場合
        elif kind == gv.G.TestLabelIndex:
            hotti = self.TestLabel[0].copy().tolist()
            for i in range(1, len(self.TestLabel)):
                hotti.extend(self.TestLabel[i].tolist())
            # End_For
            hotti = np.array(hotti, dtype=np.int32)
            return hotti[self.FlattenTestPerm]
        # 学習用画像認識実験の画像がほしい場合
        elif kind == gv.G.TrainedTestDataIndex:
            hotti = self.TrainedTestData[0].copy().tolist()
            for i in range(1, len(self.TrainedTestData)):
                hotti.extend(self.TrainedTestData[i].tolist())
            # End_For
            hotti = np.array(hotti, dtype=np.float32)
            return hotti[self.FlattenTrainedTestPerm]
        # 学習用画像認識実験の画像のラベルが欲しい場合
        elif kind == gv.G.TrainedTestLabelIndex:
            hotti = self.TrainedTestLabel[0].copy().tolist()
            for i in range(1, len(self.TrainedTestLabel)):
                hotti.extend(self.TrainedTestLabel[i].tolist())
            # End_For
            hotti = np.array(hotti, dtype=np.int32)
            return hotti[self.FlattenTrainedTestPerm]
        else:
            print("GetFlattenに予期されぬ引数が入力されました : {}".format(kind))
            exit(-1)
        # End_IfElse
    # End_Func

    # テスト画像を任意のディレクトリの画像に上書き
    def __ChangeTestDataImage(self):
        print("テスト画像上書き Path:{}".format(self.OverwriteTestDataPath))
        fileNames = np.array(
            self.__GetAllImagePath(
                self.OverwriteTestDataPath).copy())
        pm = np.random.permutation(self.TestSize)
        for i in range(len(fileNames)):
            # ランダムに並び替える
            try:
                fileNames[i] = fileNames[i][pm]
            except ValueError:
                print('Value Error.')
            # 上書きしていく
            for j in range(self.TestSize):
                img = Image.open(fileNames[i][j])
                img = img.resize((self.Width, self.Height))
                img = np.array(img, dtype=np.float32)
                """ path= の部分に{}がない? """
                print(
                    "i={} shape={} path=".format(
                        i, img.shape, fileNames[i][j]))
                img = img[:, :, :3]
                img = img.transpose(2, 0, 1)
                self.TestData[i][j] = img.copy()
            # End_For
        # End_For
    # End_Method

    # 既知画像認識実験用画像を任意のディレクトリの画像に上書き
    def __ChangeTrainedTestDataImage(self):
        print("既知画像認識実験用画像上書き Path:{}".format(
            self.OverwriteTrainedTestDataPath))
        fileNames = np.array(
            self.__GetAllImagePath(
                self.OverwriteTrainedTestDataPath).copy())
        pm = np.random.permutation(self.TestSize)
        for i in range(len(fileNames)):
            # ランダムに並び替える
            try:
                fileNames[i] = fileNames[i][pm]
            except ValueError:
                print('Value Error.')
            # 上書きしていく
            for j in range(self.TestSize):
                img = Image.open(fileNames[i][j])
                img = img.resize((self.Width, self.Height))
                img = np.array(img, dtype=np.float32)
                print(
                    "i={} shape={} path=".format(
                        i, img.shape, fileNames[i][j]))
                img = img[:, :, :3]
                img = img.transpose(2, 0, 1)
                self.TrainedTestData[i][j] = img.copy()
            # End_For
        # End_For
    # End_Method

    # 任意のディレクトリ下にあるすべての画像パスをクラスごとに取得
    def __GetAllImagePath(self, imgDir):
        dirNames = glob.glob(imgDir + "/*")
        fileNames = [[] for d in self.DirNames]
        for i in range(len(dirNames)):
            globb = [glob.glob(("{}/*" + e).format(dirNames[i]))
                     for e in self.Extentions]
            for l in range(len(globb)):
                for m in range(len(globb[l])):
                    fileNames[i].append(globb[l][m])
                # End_For
            # End_For
        # End_For
        return fileNames
    # End_Method
# End_Class
