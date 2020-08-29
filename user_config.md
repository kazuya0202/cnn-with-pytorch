# user_config

+ `(required)`がつくものは、ファイルに記載されていないとエラーになるため、残しておいてください。

### yamlファイルの特徴

+ インデント（字下げ）で階層を区別するため、初期の状態から変えないでください。
+ `'`（シングルクォーテーション）でくくると、`\`の記号をエスケープする必要はない。
  + `"`（ダブルクォーテーション）の場合は、`\\`でエスケープする必要あり。
+ `default value`から変更しない場合、コメントアウト or 消してもOK。
  + コメントアウトするには、行の先頭に`#`をつける。

<br>

## path (required)

| key        | type  | default value       | description                            |
| ---------- | ----- | ------------------- | -------------------------------------- |
| `dataset`  | `str` | `./dataset`         | データセットのパス [*1]                |
| `mistaken` | `str` | `./mistaken`        | 学習中に間違えた画像の保存先           |
| `model`    | `str` | `./`                | 学習モデルの保存先                     |
| `config`   | `str` | `./config`          | 使用した画像などの設定ファイルの保存先 |
| `log`      | `str` | `./logs`            | ログ、レートファイルの保存先           |
| `gradcam`  | `str` | `./GradCAM_results` | Grad-CAM画像の保存先                   |

**Note [*1]**：

データセットのパスは、各クラスごとにフォルダにまとめ、その親フォルダのパスを指定する。

> 例
>
> ```
> Images/                <- ここを指定
>   ├─ class_A/
>   │      ├─ 001.jpg
>   │      └─ ...
>   ├─ class_B/
>   │      ├─ 001.jpg
>   │      └─ ...
>   └─ ...
> ```

<br>

## dataset (required)

| key          | type            | default value            | description                                                |
| ------------ | --------------- | ------------------------ | ---------------------------------------------------------- |
| `limit_size` | `int`           | `-1`                     | 各データセットで使用する画像のの最大枚数（`-1`は制限なし） |
| `test_size`  | `[int | float]` | `0.1`                    | テスト画像の割合・枚数（小数→割合 / 整数→枚数）            |
| `extensions` | `List[str]`     | `["jpg", "png", "jpeg"]` | 対象画像の拡張子                                           |

<br>

## gradcam (required)

| key             | type   | default value | description                        |
| --------------- | ------ | ------------- | ---------------------------------- |
| `enabled`       | `bool` | `false`       | Grad-CAMを実行する                 |
| `only_mistaken` | `bool` | `true`        | 間違えたときだけGrad-CAMを実行する |
| `layer`         | `str`  | `conv5`       | 可視化する層                       |

<br>

## network (required)

| key                            | type   | default value | description                                  |
| ------------------------------ | ------ | ------------- | -------------------------------------------- |
| `height`                       | `int`  | `60`          | 画像の入力サイズ（高さ）                     |
| `width`                        | `int`  | `60`          | 画像の入力サイズ（幅）                       |
| `channels`                     | `int`  | `3`           | 画像のチャンネル数 [*1]                      |
| `epoch`                        | `int`  | `10`          | エポック数                                   |
| `batch`                        | `int`  | `128`         | 1バッチの画像枚数（何枚ずつ行うか）          |
| `subdivision`                  | `int`  | `4`           | バッチの細分化                               |
| `save_cycle`                   | `int`  | `0`           | 学習モデルの保存サイクル [*2]                |
| `test_cycle`                   | `int`  | `1`           | 学習モデルのテストサイクル [*2]              |
| `gpu_enabled`                  | `bool` | `true`        | GPUを使用する（GPUが使用できない場合は無視） |
| `is_save_final_model`          | `bool` | `true`        | 最終モデルを保存する                         |
| `is_shuffle_dataset_per_epoch` | `bool` | `true`        | エポック毎にデータセットをシャッフルする     |

**Note [*1]**：

カラー画像（RGB）：`3`  
グレースケール画像：`1`

**Note [*2]**：

`0`：何もしな  
`10`：10エポック毎に実行  
`N`：Nエポック毎に実行

<br>

## option (required)

| key                          | type   | default value | description                                      |
| ---------------------------- | ------ | ------------- | ------------------------------------------------ |
| `is_show_network_difinition` | `bool` | `true`        | 構築したネットワーク、クラスなどの定義を表示する |
| `is_save_debug_log`          | `bool` | `true`        | ログをファイルに保存する                         |
| `is_save_rate_log`           | `bool` | `true`        | レートをファイルに保存する                       |
| `is_available_re_training`   | `bool` | `false`       | 保存するモデルを再学習可能な状態にする [*1]      |
| `re_training`                | `bool` | `false`       | 再学習可能なモデルの学習を再開する               |
| `load_model_path`            | `str`  | `""`          | 再学習可能なモデルのパス                         |

**Note [*1]**：

再学習可能な状態のモデルはサイズが大きくなるため注意。（1GB以上）