import cnn2

net = cnn2.Net((60, 60))

print(net.features._modules['conv2'].__class__.__name__ == 'Conv2d')
