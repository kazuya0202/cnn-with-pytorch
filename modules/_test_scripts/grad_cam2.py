from pathlib import Path
from typing import List, Tuple

# import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

import cnn
import toml_settings as _tms
from grad_cam_impl import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity
)


def load_images(image_paths: List[str], size: tuple = (60, 60)):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path, size)
        images.append(image)
        raw_images.append(raw_image)

    return images, raw_images


# def get_classtable():
#     classes = []
#     with open("samples/synset_words.txt") as lines:
#         for line in lines:
#             line = line.strip().split(" ", 1)[1]
#             line = line.split(", ", 1)[0].replace(" ", "_")
#             classes.append(line)
#     return classes


def preprocess(image_path: str, input_size: tuple):
    raw_image = cv2.imread(image_path)
    # raw_image = cv2.resize(raw_image, (224,) * 2)
    raw_image = cv2.resize(raw_image, input_size)

    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(raw_image[..., ::-1].copy())

    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def main(model: cnn.Net, classes: List[str], input_size: Tuple[int, int]):

    # print("Mode:", ctx.invoked_subcommand)
    # classes = ['crossing', 'klaxon', 'noise']
    # input_size = (60, 60)

    # model = cnn.Net(input_size)
    device = next(model.parameters()).device
    # model.to(device)
    model.eval()

    image_paths = ['./recognition_datasets/Images/crossing/crossing-samp1_3_4.jpg',
                   './recognition_datasets/Images/crossing/crossing-samp1_3_3.jpg']
    images, raw_images = load_images(image_paths, input_size)
    images = torch.stack(images).to(device)

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted
    ids = ids.cpu().numpy()  # numpy

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    topk = 3
    target_layer = 'conv5'
    output_dir = Path('results')

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            name_fmt = f'{j}-' + '{}' + f'-{classes[ids[j, i]]}.png'

            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            path = output_dir.joinpath(name_fmt.format('guided')).as_posix()
            save_gradient(filename=path, gradient=gradients[j])

            # Grad-CAM
            path = Path(output_dir, name_fmt.format(f'gradcam-{target_layer}')).as_posix()
            save_gradcam(filename=path, gcam=regions[j, 0], raw_image=raw_images[j])

            # Guided Grad-CAM
            path = Path(output_dir, name_fmt.format(f'guided_gradcam-{target_layer}')).as_posix()
            save_gradient(filename=path, gradient=torch.mul(regions, gradients)[j])


def demo2(image_paths, output_dir, cuda):
    """
    Generate Grad-CAM at different layers of ResNet-152
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # The four residual layers
    target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_class = 243  # "bull mastif"

    # Images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    # ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    ids_ = torch.tensor([[target_class]] * len(images), dtype=torch.long).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            # save_gradcam(
            #     filename=osp.join(
            #         output_dir,
            #         "{}-{}-gradcam-{}-{}.png".format(
            #             j, "resnet152", target_layer, classes[target_class]
            #         ),
            #     ),
            #     gcam=regions[j, 0],
            #     raw_image=raw_images[j],
            # )


def demo3(image_paths, arch, topk, stride, n_batches, output_dir, cuda):
    """
    Generate occlusion sensitivity maps
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images, _ = load_images(image_paths)
    images = torch.stack(images).to(device)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 25, 35, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                # save_sensitivity(
                #     filename=osp.join(
                #         output_dir,
                #         "{}-{}-sensitivity-{}-{}.png".format(
                #             j, arch, p, classes[ids[j, i]]
                #         ),
                #     ),
                #     maps=sensitivity[j],
                # )


if __name__ == "__main__":

    classes = ['crossing', 'klaxon', 'noise']

    tms = _tms.factory()
    net = cnn.Net(tms.input_size)
    main(net, classes, tms.input_size)
