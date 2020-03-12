# -*- Coding: utf-8 -*-
# Packages
import datetime

# --- unused packages ---
# import os
# import glob
# import chainer
# import numpy as np
# import chainer.links as L
# import chainer.functions as F
# from chainer import Chain, Variable, cuda, optimizers
# from PIL import Image

# MyPackages


# IsOverwriteTestData を True にするとDatasetSizeすべてが学習用画像になります(DatasetSize == TrainSize)
# テストデータを上書きするときは，上書き用フォルダのフォルダ名をクラス名に統一してください

# Global Variable
class G:
    # -----------------------------各環境・実験に合わせて書き換えてください-----------------------------
    # データセット取得関係(数値は各クラス毎の枚数)
    DataSetSize = 200  # テストデータと学習データの合計
    TestSize = 5  # テストデータ枚数
    TrainSize = DataSetSize - TestSize  # 学習データ枚数
    MinibatchSize = 100
    Height = 60  # 画像サイズ(高さ，このサイズにリサイズされる)
    Width = 60  # 画像サイズ(横幅，このサイズにリサイズされる)
    EpochNum = 5  # エポック数

    # --- config for kannon
    # DataSetSize = 4464  # テストデータと学習データの合計
    # TestSize = 264  # テストデータ枚数
    # TrainSize = DataSetSize - TestSize  # 学習データ枚数
    # MinibatchSize = 105
    # Height = 80  # 画像サイズ(高さ，このサイズにリサイズされる)
    # Width = 80  # 画像サイズ(横幅，このサイズにリサイズされる)
    # EpochNum = 10  # エポック数

    # Extention = ["png", "jpg", "jpeg", "bmp", "gif"]  # 使用する画像形式
    Extention = ["jpg"]

    # パス関係
    # #使用する画像(各クラスごとにフォルダにまとめて，そのフォルダをまとめたパスを指定)
    ImagePath = r"C:\ichiya\repos\github.com\kazuya0202\cnn-with-pytorch\recognition_datasets\Images"
    # ImagePath = r".\rec\Images"
    FalsePath = r".\rec\False"
    SavePicklePath = r".\rec"

    # ImagePath = r"C:\ichiya\prcn2019\prcn2019-dev\spectrum-conv\_Recognition\datasets\Images"
    # FalsePath = r"C:\ichiya\prcn2019\prcn2019-dev\spectrum-conv\_Recognition\False"
    # SavePicklePath = r"C:\ichiya\prcn2019\prcn2019-dev\spectrum-conv\_Recognition\Pickle"

    # --- kazuya
    is_every_save_pickle = False
    # ------------

    # 実行環境
    IsGPU = True  # GPUを使う場合
    IsLinux = False  # Linuxを使う場合(\と/の変換)
    # Pycharmだとimportがガバったりするよね(True未実装，VisualStudioで実行してください)
    IsPycharm = False

    # 実行ログ関係
    IsDebugLog = True  # コンソールに表示されてる文字をテキストファイルに保存(LogPathに)
    IsRateLog = True  # 各Epoch毎の認識率を保存
    LogFilePath = r".\rec\Logs"
    RateLogFilePath = r".\rec\Logs"

    # 学習・実験関係
    IsLoadPickle = False  # 学習済みモデルを使用するか(下のPickleパスと合わせて)
    LoadPicklePath = r""  # 学習済みモデルのパス
    # 1エポックごとに認識実験を行うか(上のRateLogFilePathにcsvで記録)
    IsTestRecognitionEachEpoch = True
    # 既知画像認識実験のテスト用画像を任意の画像に上書き(OverwriteTrainedTestDataより)
    IsOverwriteTestData = False  # テスト用画像を任意の画像に上書き(OverwriteTestDataPathより)
    IsOverwriteTrainedTestData = False
    # テスト用画像を上書きする(ファイル・画像の置き方はImagePath同様)
    OverwriteTestDataPath = r"C:\ichiya\prcn2019\spectrum-conv\export\datasets\known"

    # 既知画像認識実験のテスト用画像を任意の画像に上書き(ファイル・画像の置き方はImagePath同様)
    OverwriteTrainedTestDataPath = r"C:\ichiya\prcn2019\spectrum-conv\export\datasets\unknown"

    # 学習モデル関係
    TargetLayer = "fc8"  # 出力層の名前

    # Grad-CAM関係
    GradCAM_Layer = "conv5"  # GradCAMで可視化したい層
    IsGradCAM = False  # 誤認識画像に対してGradCAMを用いるか
    # #GradCAMイメージの保存先
    GradCAM_ImagePath = r"C:\ichiya\prcn2019\_SoundRecognition\GradCAM"

    # -----------------------------↓↓↓書き換えると壊れる↓↓↓-----------------------------
    # 便利オブジェクト
    Be = None  # 画像取得用
    Model = None  # ネットワークモデル
    Optimizer = None  # オプティマイザ
    xp = None  # CPU→numpy GPU→cupy

    # GetFlatten
    TrainDataIndex = 0
    TrainLabelIndex = 1
    TestDataIndex = 2
    TestLabelIndex = 3
    AllTrainDataAndAllTrainLabel = 4
    TrainedTestDataIndex = 5
    TrainedTestLabelIndex = 6

    # ログファイル関係
    RateLogFile = None
    LogFile = None

    # ファイル名関係
    DateTimeNow = str(datetime.datetime.now().strftime(
        "ymd%Y%m%d_hms%H%M%S")).replace(" ", "_")
# End_Class
