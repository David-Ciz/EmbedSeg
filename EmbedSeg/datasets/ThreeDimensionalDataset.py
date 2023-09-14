import ast
import glob
import numpy as np
import os
import pandas as pd
import pycocotools.mask as rletools
import random
import tifffile
from scipy.ndimage import zoom
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset

from EmbedSeg.utils.generate_crops import (
    normalize_min_max_percentile,
    normalize_mean_std,
)


class ThreeDimensionalDataset(Dataset):
    """
    A class used to create a PyTorch Dataset for handling 3D
    images and label masks

    Attributes
    ----------
    image_list: list of strings containing paths to the images

    instance_list: list of strings containing paths to the GT instances

    center_image_list: list of strings containing paths to the center images

    bg_id: int
        Id corresponding to the background, default=0

    size: int
        This is set equal to a different value than `real_size`
        in case the epoch is desired to be shorter

    real_size: int
        Actual number of the number of images

    transform: PyTorch transform

    one_hot: bool, default = False
        Should be set equal to True if the GT masks are available
        in a one-hot encoded fashion
        This is not applicable in a 3D setting since any pixel
        can only be assigned to one 3D GT object
        This parameter will be deprecated for the 3D setting in a future release

    norm: str
        Should be set equal to one of `min-max-percentile`, `absolute`, `mean-std`

    type: str
        Should be set equal to one of `train`, `val` or `test`
    data_type: str
        Should be set equal to one of `8-bit` or `16-bit`

    normalization: bool
        Should be equal to True for test images and set equal to False
        during training (since crops are already normalized)
    anisotropy_factor: float
        Should be set equal to the ratio of the pixel sizes
        along the z dimension and the x dimension
        In case, the raw image is acquired such that the image
        is down-sampled along the z-diemnsion,
        then 'anisotropy_factor' should be greater than equal to 1.0

    sliced_mode: bool
        In case the 3D images should be interpreted in a 'sliced fashion'
        (i.e. a 2D model should be used for training on individual slices
        And the predictions from these slices considered independently are
         combined during inference to generate a 3D instance mask), then
        'sliced_mode' should be set equal to True

    uniform_ds_factor: int, default = 1
        If the original crops were generated by down-sampling
        in '01-data.ipynb', then the test images
        should also be down-sampled before being input to the model
        and the predictions should then be up-sampled
        to restore them back to the original size

    Methods
    -------
    __init__: Initializes an object of class `ThreeDimensionalDataset`

    __len__: Returns self.real_size if self.size = None

    convert_yx_to_cyx: Adds an additional dimension for channel

    decode_instance: In case, instances are available as tiffs,
        this method decodes them

    rle_decode: In case, instances are available as csv files (and not tiffs),
        this method decodes them
    """

    def __init__(
        self,
        data_dir="./",
        center="center-medoid",
        type="train",
        bg_id=0,
        size=None,
        transform=None,
        one_hot=False,
        norm="min-max-percentile",
        normalization=False,
        data_type="8-bit",
        anisotropy_factor=1.0,
        sliced_mode=False,
        uniform_ds_factor=1,
    ):
        print(
            "3-D `{}` dataloader created! Accessing data from {}/{}/".format(
                type, data_dir, type
            )
        )

        # get image and instance list
        image_list = glob.glob(
            os.path.join(data_dir, "{}/".format(type), "images/*.tif")
        )
        image_list.sort()
        print("Number of images in `{}` directory is {}".format(type, len(image_list)))
        self.image_list = image_list

        instance_list = glob.glob(
            os.path.join(data_dir, "{}/".format(type), "masks/*.tif")
        )
        instance_list.sort()
        print(
            "Number of instances in `{}` directory is {}".format(
                type, len(instance_list)
            )
        )
        self.instance_list = instance_list

        center_image_list = glob.glob(
            os.path.join(data_dir, "{}/".format(type), center + "/*.tif")
        )
        center_image_list.sort()
        print(
            "Number of center images in `{}` directory is {}".format(
                type, len(center_image_list)
            )
        )
        print("*************************")
        self.center_image_list = center_image_list

        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform
        self.one_hot = one_hot
        self.norm = norm
        self.data_type = data_type
        self.type = type
        self.anisotropy_factor = anisotropy_factor
        self.sliced_mode = sliced_mode
        self.normalization = normalization
        self.uniform_ds_factor = uniform_ds_factor

    def convert_zyx_to_czyx(self, im, key):
        im = im[np.newaxis, ...]  # CZYX
        return im

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # load image
        image = tifffile.imread(self.image_list[index])  # ZYX
        if self.normalization and self.norm == "min-max-percentile":
            image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1, 2))
        elif self.normalization and self.norm == "mean-std":
            image = normalize_mean_std(image)
        elif self.normalization and self.norm == "absolute":
            image = image.astype(float)
            if self.data_type == "8-bit":
                image /= 255
            elif self.data_type == "16-bit":
                image /= 65535
        if self.type == "test" and self.sliced_mode:
            image = zoom(image, (self.anisotropy_factor, 1, 1), order=0)
        elif self.type == "test" and not self.sliced_mode:
            image = image[
                :: self.uniform_ds_factor,
                :: self.uniform_ds_factor,
                :: self.uniform_ds_factor,
            ]
        image = self.convert_zyx_to_czyx(image, key="image")  # CZYX
        sample["image"] = image  # CZYX
        sample["im_name"] = self.image_list[index]
        if len(self.instance_list) != 0:
            if self.instance_list[index][-3:] == "csv":
                instance = self.rle_decode(
                    self.instance_list[index], one_hot=self.one_hot
                )  # ZYX
            else:
                instance = tifffile.imread(self.instance_list[index])  # ZYX

            instance, label = self.decode_instance(instance, self.one_hot, self.bg_id)
            if self.type == "test" and self.sliced_mode:
                instance = zoom(instance, (self.anisotropy_factor, 1, 1), order=0)
                label = zoom(label, (self.anisotropy_factor, 1, 1), order=0)
            instance = self.convert_zyx_to_czyx(instance, key="instance")  # CZYX
            label = self.convert_zyx_to_czyx(label, key="label")  # CZYX
            sample["instance"] = instance
            sample["label"] = label
        if len(self.center_image_list) != 0:
            if self.center_image_list[index][-3:] == "csv":
                center_image = self.rle_decode(
                    self.center_image_list[index], center=True
                )
            else:
                center_image = tifffile.imread(self.center_image_list[index])
            if self.type == "test" and self.sliced_mode:
                center_image = zoom(
                    center_image, (self.anisotropy_factor, 1, 1), order=0
                )
            center_image = self.convert_zyx_to_czyx(
                center_image, key="center_image"
            )  # CZYX
            sample["center_image"] = center_image

        # transform
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    @classmethod
    def decode_instance(cls, pic, one_hot, bg_id=None):
        pic = np.array(pic, copy=False, dtype=np.uint16)
        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1], pic.shape[2]), dtype=np.int16
        )
        class_map = np.zeros((pic.shape[0], pic.shape[1], pic.shape[2]), dtype=np.uint8)

        if bg_id is not None:
            mask = pic > bg_id

            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(pic[mask])

                instance_map[mask] = ids
                if one_hot:
                    class_map[np.max(mask, axis=0)] = 1
                else:
                    class_map[mask] = 1

        return instance_map, class_map

    @classmethod
    def rle_decode(cls, filename, one_hot=False, center=False):
        df = pd.read_csv(filename, header=None)
        df_numpy = df.to_numpy()
        d = {}

        if one_hot:
            mask_decoded = []
            for row in df_numpy:
                d["counts"] = ast.literal_eval(row[1])
                d["size"] = ast.literal_eval(row[2])
                mask = rletools.decode(d)  # returns binary mask
                mask_decoded.append(mask)
        else:
            if center:
                mask_decoded = np.zeros(
                    ast.literal_eval(df_numpy[0][2]), dtype=np.bool
                )  # obtain size by reading first row of csv file
            else:
                mask_decoded = np.zeros(
                    ast.literal_eval(df_numpy[0][2]), dtype=np.uint16
                )  # obtain size by reading first row of csv file
            for row in df_numpy:
                d["counts"] = ast.literal_eval(row[1])
                d["size"] = ast.literal_eval(row[2])
                mask = rletools.decode(d)  # returns binary mask
                z, y, x = np.where(mask == 1)
                mask_decoded[z, y, x] = int(row[0])
        return np.asarray(mask_decoded)
