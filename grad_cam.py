from dataclasses import dataclass
from typing import List, Tuple, Union

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm

# my packages
import cnn
from grad_cam_impl import (BackPropagation, Deconvnet, GradCAM,
                           GuidedBackPropagation)


def preprocess(image_path: str, input_size: tuple) -> Tuple[Tensor, Tensor]:
    r"""Load image, convert its type to torch.Tensor and return it.

    Args:
        image_path (str): path of image.
        input_size (tuple): input image size.

    Returns:
        Tuple: image data.
    """

    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, input_size)

    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(raw_image[..., ::-1].copy())

    return image, raw_image
# end of [function] preprocess


def load_images(image_paths: List[str], input_size: tuple = (60, 60)) \
        -> Tuple[List[Tensor], List[Tensor]]:
    r"""Load images.

    Returns:
        Tuple: image data.
    """

    images = []
    raw_images = []

    for image_path in image_paths:
        image, raw_image = preprocess(image_path, input_size)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images
# end of [function] load_images


def get_data_of_gradient(gradient: Tensor) -> Tensor:
    r"""Returns gradient data.

    Args:
        gradient (Tensor): gradient.

    Returns:
        Tensor: calculated gradient data.
    """
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0

    gradient = np.uint8(gradient)
    return gradient
# end of [function] get_gradient_data


def get_data_of_grad_cam(gcam: Tensor, raw_image: Tensor,
                         paper_cmap: bool = False) -> Tensor:
    r"""Returns Grad-CAM data.

    Args:
        gcam (Tensor): Grad-CAM data.
        raw_image (Tensor): raw image data.
        paper_cmap (bool, optional): cmap. Defaults to False.

    Returns:
        Tensor: [description]
    """
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (torch.tensor(1) - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.to(np.float)) / 2

    gcam = np.uint8(gcam)
    return gcam
# end of [function] get_grad_cam_data


@dataclass()
class ExecuteGradCAM:
    def __init__(self, classes: List[str], input_size: Tuple[int, int],
                 target_layer: str, **options) -> None:
        r"""
        Args:
            classes (List[str]): classes of model.
            input_size (Tuple[int, int]): input image size.
            target_layer (str): grad cam layer.

        **options
            is_vanilla (bool): execute `Vanilla`. Defaults to False.
            is_deconv (bool): execute `Deconv Net`. Defaults to False.
            is_grad_cam (bool): execute `Grad-CAM`. Defaults to False.
        """

        self.classes = classes
        self.input_size = input_size
        self.target_layer = target_layer

        self.is_vanilla = options.pop('is_vanilla', False)
        self.is_deconv = options.pop('is_deconv', False)
        self.is_grad_cam = options.pop('is_grad_cam', False)

        self.class_num = len(classes)

        self.processed_data = {
            'vanilla': [],
            'deconv': [],
            'gbp': [],
            'gcam': [],
            'ggcam': [],
        }
    # end of [function] __init__

    @torch.enable_grad()  # enable gradient
    def main(self, model: cnn.Net, image_path: Union[list, str]) -> dict:
        """Switch execute function.

        Args:
            model (cnn.Net): model.
            image_path (Union[list, str]): path of image.

        Returns:
            List: processed image data.
        """

        model.eval()  # switch to eval

        # process one image.
        if isinstance(image_path, str):
            self._execute_one_image(model, image_path)

        # process multi images.
        elif isinstance(image_path, list):
            self._execute_multi_images(model, image_path)

        return self.processed_data
    # end of [function] main

    def _execute_one_image(self, model: cnn.Net, image_path: str) -> None:
        """Process one image.

        Args:
            model (cnn.Net): model.
            image_path (str): path of image.
        """

        device = next(model.parameters()).device  # get device

        image, raw_image = preprocess(image_path, self.input_size)
        image = image.unsqueeze_(0).to(device)

        # --- Vanilla Backpropagation ---
        bp = BackPropagation(model=model)
        _, ids = bp.forward(image)  # sorted

        # --- Deconvolution ---
        deconv = None

        if self.is_deconv:
            deconv = Deconvnet(model=model)
            _ = deconv.forward(image)

        # --- Grad-CAM / Guided Backpropagation / Guided Grad-CAM ---
        gcam = None
        gbp = None

        if self.is_grad_cam:
            gcam = GradCAM(model=model)
            _ = gcam.forward(image)

            gbp = GuidedBackPropagation(model=model)
            _ = gbp.forward(image)

        pbar = tqdm(range(self.class_num), total=self.class_num, ncols=100,
                    bar_format='{l_bar}{bar:30}{r_bar}', leave=False)
        pbar.set_description('Grad-CAM')

        for i in pbar:
            if self.is_vanilla:
                bp.backward(ids=ids[:, [i]])
                gradients = bp.generate()

                # append
                data = get_data_of_gradient(gradients)
                self.processed_data['vanilla'].append(data)

            if self.is_deconv:
                deconv.backward(ids=ids[:, [i]])
                gradients = deconv.generate()

                # append
                data = get_data_of_gradient(gradients)
                self.processed_data['deconv'].append(data)

            # Grad-CAM / Guided Grad-CAM / Guided Backpropagation
            if self.is_grad_cam:
                gbp.backward(ids=ids[:, [i]])
                gradients = gbp.generate()

                # Grad-CAM
                gcam.backward(ids=ids[:, [i]])
                regions = gcam.generate(target_layer=self.target_layer)

                # append
                data = get_data_of_gradient(gradients)
                self.processed_data['gbp'].append(data)

                data = get_data_of_grad_cam(regions[0], raw_image)
                self.processed_data['gcam'].append(data)

                data = get_data_of_gradient(torch.mul(regions, gradients))
                self.processed_data['ggcam'].append(data)

        # Remove all the hook function in the 'model'
        bp.remove_hook()

        if self.is_deconv:
            deconv.remove_hook()

        if self.is_grad_cam:
            gcam.remove_hook()
            gbp.remove_hook()
    # end of [function] _execute_one_image

    def _execute_multi_images(self, model: cnn.Net, image_paths: List[str]) -> None:
        r"""Process multiple images.

        Args:
            model (cnn.Net): model.
            image_paths (List[str]): path of images.
        """

        device = next(model.parameters()).device  # get device

        images, raw_images = load_images(image_paths, self.input_size)
        images = torch.stack(images).to(device)

        # --- Vanilla Backpropagation ---
        bp = BackPropagation(model=model)
        _, ids = bp.forward(images)  # sorted

        # --- Deconvolution ---
        deconv = None

        if self.is_deconv:
            deconv = Deconvnet(model=model)
            _ = deconv.forward(images)

        # --- Grad-CAM / Guided Backpropagation / Guided Grad-CAM ---
        gcam = None
        gbp = None

        if self.is_grad_cam:
            gcam = GradCAM(model=model)
            _ = gcam.forward(images)

            gbp = GuidedBackPropagation(model=model)
            _ = gbp.forward(images)

        pbar = tqdm(range(self.class_num), total=self.class_num, ncols=100,
                    bar_format='{l_bar}{bar:30}{r_bar}', leave=False)
        pbar.set_description('Grad-CAM')

        for i in pbar:
            if self.is_vanilla:
                bp.backward(ids=ids[:, [i]])
                gradients = bp.generate()

                # Save results as image files
                for j in range(len(images)):
                    # append
                    data = get_data_of_gradient(gradients[j])
                    self.processed_data['vanilla'].append(data)

            if self.is_deconv:
                deconv.backward(ids=ids[:, [i]])
                gradients = deconv.generate()

                for j in range(len(images)):
                    # append
                    data = get_data_of_gradient(gradients[j])
                    self.processed_data['deconv'].append(data)

            # Grad-CAM / Guided Grad-CAM / Guided Backpropagation
            if self.is_grad_cam:
                gbp.backward(ids=ids[:, [i]])
                gradients = gbp.generate()

                # Grad-CAM
                gcam.backward(ids=ids[:, [i]])
                regions = gcam.generate(target_layer=self.target_layer)

                for j in range(len(images)):
                    # append
                    data = get_data_of_gradient(gradients[j])
                    self.processed_data['gbp'].append(data)

                    data = get_data_of_grad_cam(regions[j, 0], raw_images[j])
                    self.processed_data['gcam'].append(data)

                    data = get_data_of_gradient(torch.mul(regions, gradients)[j])
                    self.processed_data['ggcam'].append(data)

        # Remove all the hook function in the 'model'
        bp.remove_hook()

        if self.is_deconv:
            deconv.remove_hook()

        if self.is_grad_cam:
            gcam.remove_hook()
            gbp.remove_hook()
    # end of [function] _execute_multi_images
# end of [class] ExecuteGradCAM
