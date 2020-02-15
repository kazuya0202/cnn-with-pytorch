## WANT TO DO

+ RuntimeWarning: invalid value encountered in less
  xa[xa < 0] = -1
  + https://teratail.com/questions/190718


+ torchaudio + nnAudio -> spectrogram
  + or torchaudioのみ
+ tensorboard
  + 学習曲線の可視化
  + loss ...
  + その他もろもろ
+ Grad CAM with pytorch
  + https://github.com/jacobgil/pytorch-grad-cam
+ テストの実行中は、...を /-\- の順にやってくやつにする
  + print('/', end='\r') yield
  + print('-', end='\r') yield
  + print('\\', end='\r') yield
  + print('-', end='\r') yield
## TODO

+ batch, image_size

image_sizeが80は batch_size が50のときで限界？
↑CNNのネットワークのLinearを半分にしたらやっと動く程度

CNNのネットワークをそのまま使うんであれば、
image: 60, batch: 100が限界

---



+ nn.Linear()

```
# X: previous nn.Linear()
# => x = F.relu(self.convN(x))
#    print(x.size())  # <- this (index(2), index(3))
#    x.view(-1, self.num_flat_features(x))
#    x = F.relu(self.convN(x))

nn.Linear(out_channels * X.height, X.width)
```

+ input image size (default network)

```
size: 32 -> torch.Size([2, 16, 6, 6])
size: 64 -> torch.Size([2, 16, 14, 14])
size: 96 -> torch.Size([2, 16, 22, 22])
size: 128 -> torch.Size([2, 16, 30, 30])

size: 60 -> torch.Size([2, 16, 13, 13])
size: 80 -> torch.Size([2, 16, 18, 18])
size: 100 -> torch.Size([2, 16, 23, 23])
```

+ input image size (okabe network)

```
size: 32 -> torch.Size([2, 256, 8, 8])
size: 64 -> torch.Size([2, 256, 16, 16])
size: 96 -> torch.Size([2, 256, 24, 24])
size: 128 -> torch.Size([2, 256, 32, 32])

size: 60 -> torch.Size([2, 256, 15, 15])
size: 80 -> torch.Size([2, 256, 20, 20])
size: 100 -> torch.Size([2, 256, 25, 25])
```
