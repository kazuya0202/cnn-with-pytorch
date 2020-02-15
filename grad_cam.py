from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

# import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

import cnn
from grad_cam_impl import (BackPropagation, Deconvnet, GradCAM,
                           GuidedBackPropagation)


def preprocess(image_path: str, input_size: tuple):
    raw_image = cv2.imread(image_path)
    # raw_image = cv2.resize(raw_image, (224,) * 2)
    raw_image = cv2.resize(raw_image, input_size)

    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(raw_image[..., ::-1].copy())

    return image, raw_image


def load_images(image_paths: List[str], input_size: tuple = (60, 60)):
    images = []
    raw_images = []
    # print("Images:")
    for i, image_path in enumerate(image_paths):
        # print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path, input_size)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def load_image(image_path: str, input_size: tuple = (60, 60)):
    image, raw_image = preprocess(image_path, input_size)
    return image, raw_image


def get_gradient_data(gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0

    gradient = np.uint8(gradient)
    return gradient


def save_gradient(filename, gradient):
    # gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    # gradient -= gradient.min()
    # gradient /= gradient.max()
    # gradient *= 255.0
    # cv2.imwrite(filename, np.uint8(gradient))

    gradient = get_gradient_data(gradient)
    cv2.imwrite(filename, gradient)


def get_gradcam_data(gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2

    gcam = np.uint8(gcam)
    return gcam


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    # gcam = gcam.cpu().numpy()
    # cmap = cm.jet_r(gcam)[..., :3] * 255.0
    # if paper_cmap:
    #     alpha = gcam[..., None]
    #     gcam = alpha * cmap + (1 - alpha) * raw_image
    # else:
    #     gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    # cv2.imwrite(filename, np.uint8(gcam))

    gcam = get_gradcam_data(gcam, raw_image, paper_cmap)
    cv2.imwrite(filename, gcam)


@dataclass()
class ExecuteGradCam:
    classes: List[str]  # classes of model
    input_size: Tuple[int, int]  # input image size
    target_layer: str
    save_dir: str  # save directory

    is_vanilla: bool = False
    is_deconv: bool = False
    is_gradcam: bool = True

    def main(self, model: cnn.Net, image_path: Union[list, str]):
        if isinstance(image_path, str):
            image_path = [image_path]

        ret_data = self.__execute(model, image_path)

        return ret_data

    def __execute(self, model: cnn.Net, image_paths: List[str]):
        processed_data = {
            'vanilla': [],
            'deconv': [],
            'gbp': [],
            'gcam': [],
            'ggcam': [],
        }

        device = next(model.parameters()).device
        model.eval()

        images, raw_images = load_images(image_paths, self.input_size)
        images = torch.stack(images).to(device)

        cls_num = len(self.classes)
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- Vanilla Backpropagation ---
        bp = BackPropagation(model=model)
        probs, ids = bp.forward(images)  # sorted

        # --- Deconvolution ---
        deconv = None

        if self.is_deconv:
            deconv = Deconvnet(model=model)
            _ = deconv.forward(images)

        # --- Grad-CAM / Guided Backpropagation / Guided Grad-CAM ---
        gcam = None
        gbp = None

        if self.is_gradcam:
            gcam = GradCAM(model=model)
            _ = gcam.forward(images)

            gbp = GuidedBackPropagation(model=model)
            _ = gbp.forward(images)

        # probs = probs.detach().cpu().numpy()  # to numpy
        # ids_np = ids.detach().cpu().numpy()  # to numpy

        pbar = tqdm(range(cls_num), total=cls_num, ncols=100,
                    bar_format='{l_bar}{bar:30}{r_bar}', leave=False)

        for i in pbar:
            if self.is_vanilla:
                bp.backward(ids=ids[:, [i]])
                gradients = bp.generate()

                # Save results as image files
                for j in range(len(images)):
                    # fmt = '%d-{}-%s.png' % (j, self.classes[ids_np[j, i]])
                    # print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                    # append
                    _grad = get_gradient_data(gradients[j])
                    processed_data['vanilla'].append(_grad)

                    # save as image
                    # _p = save_dir.joinpath(fmt.format('vanilla'))
                    # save_gradient(str(_p), gradients[j])

            if self.is_deconv:
                deconv.backward(ids=ids[:, [i]])
                gradients = deconv.generate()

                for j in range(len(images)):
                    # fmt = '%d-{}-%s.png' % (j, self.classes[ids_np[j, i]])
                    # print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                    # append
                    _grad = get_gradient_data(gradients[j])
                    processed_data['deconv'].append(_grad)

                    # save as image
                    # _p = save_dir.joinpath(fmt.format('deconvnet'))
                    # save_gradient(str(_p), gradients[j])

            # Grad-CAM / Guided Grad-CAM / Guided Backpropagation
            if self.is_gradcam:
                gbp.backward(ids=ids[:, [i]])
                gradients = gbp.generate()

                # Grad-CAM
                gcam.backward(ids=ids[:, [i]])
                regions = gcam.generate(target_layer=self.target_layer)

                for j in range(len(images)):
                    # fmt = '%d-{}-%s.png' % (j, self.classes[ids_np[j, i]])
                    # print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                    # append
                    _grad = get_gradient_data(gradients[j])
                    processed_data['gbp'].append(_grad)

                    _grad = get_gradcam_data(regions[j, 0], raw_images[j])
                    processed_data['gcam'].append(_grad)

                    _grad = get_gradient_data(torch.mul(regions, gradients)[j])
                    processed_data['ggcam'].append(_grad)

                    # save as image - Guided Backpropagation
                    # _p = save_dir.joinpath(fmt.format('guided-bp'))
                    # save_gradient(str(_p), gradients[j])

                    # save as image - Grad-CAM
                    # _p = save_dir.joinpath(fmt.format(f'gradcam-{self.target_layer}'))
                    # save_gradcam(str(_p), regions[j, 0], raw_images[j])

                    # save as image - Guided Grad-CAM
                    # _p = save_dir.joinpath(fmt.format(f'guided_gradcam-{self.target_layer}'))
                    # save_gradient(str(_p), torch.mul(regions, gradients)[j])

        # Remove all the hook function in the 'model'
        bp.remove_hook()

        if self.is_deconv:
            deconv.remove_hook()

        if self.is_gradcam:
            gcam.remove_hook()
            gbp.remove_hook()

        return processed_data
