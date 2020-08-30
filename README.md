# CNN implemented PyTorch with Grad-CAM

|      Script      |   Description    |
| :--------------: | :--------------: |
|     main.py      |     学習実行     |
| user_config.yaml |   ユーザー設定   |
|    _valid.py     | 学習モデルの検証 |

<br>

## Requirements

+ `pip` モジュール

  1. PyTorch

     PyTorch インストール。

  2. その他モジュール

     ```sh
     $ pip install -r requirements.txt
     ```

+ CUDA / cuDNN

  [【Windows】CUDA-cuDNN インストール](https://ichiya.netlify.com/posts/2020/02/29/_20200229.html) 参照。

<br>

## Usage

`user_config.yaml`は`main.py`と同じディレクトリに置く。

もしくは、実行時にパスを指定する。

---

1. *[user_config.yaml](https://github.com/kazuya0202/cnn-with-pytorch/blob/dev/user_config.yaml)* を編集する `Notes: 1`

   ※ 変数名は変更しない。

2. スクリプトを実行する `Notes: 2`

   + `user_config.yaml`が同じディレクトリにない場合、もしくは、別名のファイルを指定する場合`--path`オプションで直接指定する。
   
   ```sh
   $ python main.py
   # or
   $ python main.py --path <yaml path>
   ```

<br>

## Notes

### 【1】

+ `user-settings.toml`の記述が正しくない場合、以下のようなエラーが出る。

<details><summary>出力例（クリックして展開）</summary>


```sh
$ python main.py

Traceback (most recent call last):
  File "D:\scoop\kazuya\apps\python\3.7.4\lib\site-packages\toml\decoder.py", line 456, in loads
    multibackslash)
  File "D:\scoop\kazuya\apps\python\3.7.4\lib\site-packages\toml\decoder.py", line 725, in load_line
    value, vtype = self.load_value(pair[1], strictly_valid)
  File "D:\scoop\kazuya\apps\python\3.7.4\lib\site-packages\toml\decoder.py", line 840, in load_value
    v = int(v, 0)
ValueError: invalid literal for int() with base 0: 'tru'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "main.py", line 234, in <module>
    main.execute()
  File "main.py", line 22, in execute
    tms = _tms.factory()
  File "D:\workspace\repos\github.com\kazuya0202\cnn-with-pytorch\toml_settings.py", line 41, in factory
    usr_toml = toml.load('./user_settings.toml')
  File "D:\scoop\kazuya\apps\python\3.7.4\lib\site-packages\toml\decoder.py", line 112, in load
    return loads(ffile.read(), _dict, decoder)
  File "D:\scoop\kazuya\apps\python\3.7.4\lib\site-packages\toml\decoder.py", line 458, in loads
    raise TomlDecodeError(str(err), original, pos)
toml.decoder.TomlDecodeError: invalid literal for int() with base 0: 'tru' (line 83 column 1 char 1736)
```

</details>

<br>

### 【2】

+ GPUのメモリ不足で *CUDA Memory Error* が出る場合、`user_config.yaml`の`subdivision`の値を増やす。
+ テスト時に間違えた画像のみ**Grad-CAM**を行う。
+ クラス分類は、自動で定義される。
  + `[dataset] > dataset_path`の子ディレクトリをクラスとして扱う。

