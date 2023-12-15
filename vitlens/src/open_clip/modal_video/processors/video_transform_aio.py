import sys
import math
import warnings
import numbers
import random
import numpy as np
import PIL
import skimage
import skimage.transform
import torchvision
import torch
from open_clip.modal_video.processors import functional_aio as F
from torchvision import transforms
import torchvision.transforms.functional as FF
from PIL import Image


def convert_img(img):
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format"""
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img


class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip_test (list of numpy.ndarray): clip_test (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, "Got {0} instead of 3 channels".format(ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image\
            but got list of {0}".format(
                    type(clip[0])
                )
            )

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError(
                    "Expected numpy.ndarray or PIL.Image\
                but got list of {0}".format(
                        type(clip[0])
                    )
                )
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img
        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = tensor_clip.div(255)
            return tensor_clip


class ToTensor(object):
    """Converts numpy array to tensor"""

    def __call__(self, array):
        tensor = torch.from_numpy(array)
        return tensor


class ColorDistortion(object):
    def __init__(self, s=1.0):
        self.s = s
        self.color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.rnd_color_jitter = transforms.RandomApply([self.color_jitter], p=0.8)
        self.rnd_gray = transforms.RandomGrayscale(p=0.2)

    def __call__(self, video):
        color_distort = transforms.Compose([self.rnd_color_jitter, self.rnd_gray])
        return color_distort(video)


class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly
    with a probability 0.5
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip_test
        """
        if random.random() < self.p:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip]
            else:
                raise TypeError(
                    "Expected numpy.ndarray or PIL.Image"
                    + " but got list of {0}".format(type(clip[0]))
                )
        return clip


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation="nearest"):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = F.resize_clip(clip, new_size, interpolation=self.interpolation)
        return resized


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation="nearest"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = F.resize_clip(clip, self.size, interpolation=self.interpolation)
        return resized


class RandomCrop(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )
        if w > im_w or h > im_h:
            error_msg = (
                "Initial image size should be larger then "
                "cropped size but got cropped sizes : ({w}, {h}) while "
                "initial image is ({im_w}, {im_h})".format(
                    im_w=im_w, im_h=im_h, w=w, h=h
                )
            )
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class CornerCrop(object):
    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ["c", "tl", "tr", "bl", "br"]

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        corner_imgs = list()
        for n in self.crop_positions:
            # print(n)
            if n == "c":
                th, tw = (self.size, self.size)
                x1 = int(round((w - tw) / 2.0))
                y1 = int(round((h - th) / 2.0))
                x2 = x1 + tw
                y2 = y1 + th
            elif n == "tl":
                x1 = 0
                y1 = 0
                x2 = self.size
                y2 = self.size
            elif n == "tr":
                x1 = w - self.size
                y1 = 0
                x2 = w
                y2 = self.size
            elif n == "bl":
                x1 = 0
                y1 = h - self.size
                x2 = self.size
                y2 = h
            elif n == "br":
                x1 = w - self.size
                y1 = h - self.size
                x2 = w
                y2 = h
            corner_imgs.append(imgs[:, y1:y2, x1:x2, :])
        return corner_imgs

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[
                random.randint(0, len(self.crop_positions) - 1)
            ]


class RandomRotation(object):
    """Rotate entire clip_test randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number," "must be positive")
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence," "it must be of len 2.")

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )

        return rotated


class STA_RandomRotation(object):
    """Rotate entire clip_test randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number," "must be positive")
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence," "it must be of len 2.")

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        bsz = len(clip)
        angle = random.uniform(self.degrees[0], self.degrees[1])
        angles = [(i + 1) / (bsz + 1) * angle for i in range(bsz)]
        if isinstance(clip[0], np.ndarray):
            rotated = [
                skimage.transform.rotate(img, angles[i]) for i, img in enumerate(clip)
            ]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angles[i]) for i, img in enumerate(clip)]
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )

        return rotated


class Each_RandomRotation(object):
    """Rotate entire clip_test randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number," "must be positive")
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence," "it must be of len 2.")

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        bsz = len(clip)
        angles = [random.uniform(self.degrees[0], self.degrees[1]) for i in range(bsz)]
        # print(angles)
        if isinstance(clip[0], np.ndarray):
            rotated = [
                skimage.transform.rotate(img, angles[i]) for i, img in enumerate(clip)
            ]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angles[i]) for i, img in enumerate(clip)]
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )

        return rotated


class CenterCrop(object):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )
        if w > im_w or h > im_h:
            error_msg = (
                "Initial image size should be larger then "
                "cropped size but got cropped sizes : ({w}, {h}) while "
                "initial image is ({im_w}, {im_h})".format(
                    im_w=im_w, im_h=im_h, w=w, h=h
                )
            )
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.0))
        y1 = int(round((im_h - h) / 2.0))
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip_test
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip_test (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError("Color jitter not yet implemented for numpy arrays")
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )

            # Create img sync_dir function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(
                        img, brightness
                    )
                )
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(
                        img, saturation
                    )
                )
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue)
                )
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(
                        img, contrast
                    )
                )
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )
        return jittered_clip


class EachColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip_test
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip_test (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError("Color jitter not yet implemented for numpy arrays")
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )

            # Create img sync_dir function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(
                        img, brightness
                    )
                )
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(
                        img, saturation
                    )
                )
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue)
                )
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(
                        img, contrast
                    )
                )
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )
        return jittered_clip


class Normalize(object):
    """Normalize a clip_test with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this sync_dir
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This sync_dir acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Tensor clip_test of size (T, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor clip_test.
        """
        return F.normalize(clip, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class TensorToNumpy(object):
    def __init__(self):
        print("convert to numpy")

    def __call__(self, clip):
        np_clip = clip.permute(1, 2, 3, 0).cpu().detach().numpy()
        pil_clip = [
            Image.fromarray(np.uint8(numpy_image)).convert("RGB")
            for numpy_image in np_clip
        ]
        return pil_clip


class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


# wx: add for MVM
_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


class GroupRandomResizedCropAndInterpolationWithTwoClips:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        second_size=None,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
        second_interpolation="lanczos",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.second_interpolation = (
            _pil_interp(second_interpolation)
            if second_interpolation is not None
            else None
        )
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(imglist, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): list of Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        img = imglist[0]
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, imglist):
        """
        Args:
            imglist (PIL Image): list of Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(imglist, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return [
                FF.resized_crop(img, i, j, h, w, self.size, interpolation)
                for img in imglist
            ]
        else:
            return [
                FF.resized_crop(img, i, j, h, w, self.size, interpolation)
                for img in imglist
            ], [
                FF.resized_crop(
                    img, i, j, h, w, self.second_size, self.second_interpolation
                )
                for img in imglist
            ]

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [_pil_interpolation_to_str[x] for x in self.interpolation]
            )
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0}".format(interpolate_str)
        if self.second_size is not None:
            format_string += ", second_size={0}".format(self.second_size)
            format_string += ", second_interpolation={0}".format(
                _pil_interpolation_to_str[self.second_interpolation]
            )
        format_string += ")"
        return format_string


class GroupCenterCropAndResizedWithTwoClips:
    """Centercrop the given PIL Image and the resize to two sizes
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        resize_size=512,
        centercrop_size=384,
        size=None,
        second_size=None,
        interpolation="bilinear",
        second_interpolation="lanczos",
    ):
        if isinstance(centercrop_size, tuple):
            self.centercrop_size = centercrop_size
        else:
            self.centercrop_size = (centercrop_size, centercrop_size)
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.second_interpolation = _pil_interp(second_interpolation)

        self.pre_resize = Resize(resize_size)
        self.center_crop = CenterCrop(size=self.centercrop_size)

    def __call__(self, imglist):
        """
        Args:
            imglist (PIL Image): list of Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        pre_resize_imglist = self.pre_resize(imglist)
        center_crop_images = self.center_crop(pre_resize_imglist)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return [
                FF.resize(img, self.size, interpolation) for img in center_crop_images
            ]
        else:
            return [
                FF.resize(img, self.size, interpolation) for img in center_crop_images
            ], [
                FF.resize(img, self.second_size, self.second_interpolation)
                for img in center_crop_images
            ]


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == "L":
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == "RGB":
            if self.roll:
                return np.concatenate(
                    [np.array(x)[:, :, ::-1] for x in img_group], axis=2
                )
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]"""

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255.0) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_pre_norm):
        tensor = tensor_pre_norm
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor