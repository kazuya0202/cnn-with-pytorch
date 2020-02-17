# CNN implemented PyTorch with Grad-CAM

## Requirements

+ `pip` モジュール

  1. PyTorch

     [PyTorch インストール方法]() 参照。

  2. その他モジュール

     ```sh
     $ pip install -r requirements.txt
     ```

+ CUDA / cuDNN

  [【Windows】CUDA / cuDNN インストール]() 参照。

<br>

## Usage

1. `user-settings.toml`を編集する[^1]

   + データセットのパス
   + エポック数　...

2. スクリプトを実行する[^2]

   ```sh
   $ python main.py
   ```

<br>

## Notes

[^1]: `user-settings.toml`の記述が正しくない場合、以下のようなエラーが出る。

<div><summary>クリックして展開</summary><details>

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

</details></div>

[^2]: GPUのメモリ不足で *CUDA Memory Error* が出る場合、`user_settings.toml`の`subdivision`の値を増やす。
