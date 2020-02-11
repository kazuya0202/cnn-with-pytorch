# my packages
import torch_utils as tu
import global_variables as _gv


def main():
    gv = _gv.GlobalVariables()
    pth_path = './recognition_datasets/0_2020Feb10_21h48m40s_final.pth'

    vm = tu.ValidModel(pth_path, use_gpu=gv.use_gpu)
    # pred_label = vm.valid('./recognition_datasets/Images/crossing/crossing-samp1_9_9.jpg')
    # print(pred_label)

    for x in open('config/known_used_images.txt').readlines():
        predicted: tu.PredictedResult = vm.valid(x.strip())
        print(predicted.label)

    return 0


if __name__ == '__main__':
    exit(main())
