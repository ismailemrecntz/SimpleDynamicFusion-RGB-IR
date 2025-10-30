import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2



import os
from typing import Tuple, List
import numpy as np
from PIL import Image

import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# --- Statistics ---

RGB_MEAN = [0.23270203, 0.26730181, 0.23411764]
RGB_STD  = [0.17052584, 0.17253114, 0.17188749]
IR_MEAN  = [0.3615869989505565]
IR_STD   = [0.07645516018109451]


class FusionModelDataset(Dataset):
    """
    images/<name>.png  ->  4 channels (RGB + IR) (H,W,4)
    labels/<name>.png  ->  1 channel class mask (H,W)

    Outputs (train/val):
        rgb  : Tensor, (3, 480, 640)
        ir   : Tensor, (1, 480, 640)
        mask : LongTensor, (480, 640)

    Outputs (test):
        rgb  : Tensor, (3, 480, 640)
        ir   : Tensor, (1, 480, 640)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        have_label: bool = True,
        rgb_size: Tuple[int, int] = (480, 640),
        ir_size: Tuple[int, int] = (480, 640),
        ignore_index: int = 255,
        use_weather_aug: bool = False,
    ):
        super().__init__()

        assert split in ["train", "val", "test" ,"test_day","test_night"], 'split must be "train"|"val"|"test"'
        self.data_dir = data_dir
        self.split = split
        self.have_label = have_label
        self.rgb_size = rgb_size
        self.ir_size = ir_size
        self.ignore_index = ignore_index

        # isim listesi
        with open(os.path.join(data_dir, f"{split}.txt"), "r") as f:
            self.names: List[str] = [line.strip() for line in f if line.strip()]
        self.n_data = len(self.names)

        # --------- Transforms ---------
        # 1) Common geometric transforms (RGB, IR, mask same transform)
        if split == "train":
            self.shared_geom = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.0),
                    A.ShiftScaleRotate(
                        shift_limit=0.10,
                        scale_limit=0.10,
                        rotate_limit=10,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=self.ignore_index,
                        p=0.3, 
                    ),
                ],
                additional_targets={"ir": "image"},
            )
        else:
            # There are no geometric augmentations for val/test
            self.shared_geom = A.Compose([], additional_targets={"ir": "image"})

        # 2) RGB+mask 
        self.resize_rgb_mask = A.Compose(
            [A.Resize(self.rgb_size[0], self.rgb_size[1], interpolation=cv2.INTER_LINEAR)]
        )

        # 3) IR 
        self.resize_ir = A.Compose(
            [A.Resize(self.ir_size[0], self.ir_size[1], interpolation=cv2.INTER_LINEAR)]
        )

        self.resize_label = A.Compose(
            [A.Resize(self.ir_size[0], self.ir_size[1], interpolation=cv2.INTER_NEAREST)]
        )

        # 4) RGB normalize + tensor
        rgb_aug = []
        if split == "train":
            rgb_aug + [
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                ], p=0.1),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3
                ), 
                A.Normalize(mean=RGB_MEAN, std=RGB_STD),
                ToTensorV2()
            ]
            if use_weather_aug:
                rgb_aug += [
                    A.OneOf(
                        [
                            A.RandomRain(p=1),
                            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1),
                            A.RandomSunFlare(p=1),
                        ],
                        p=0.1,
                    )
                ]
        rgb_aug += [A.Normalize(mean=RGB_MEAN, std=RGB_STD), ToTensorV2()]
        self.rgb_only = A.Compose(rgb_aug)

        # 5) IR normalize + tensor
        self.ir_only = A.Compose([A.Normalize(mean=IR_MEAN, std=IR_STD), ToTensorV2()])

    # ---------- IO ----------
    def _read_image_4ch(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, "images", f"{name}.png")
        img = np.asarray(Image.open(path))
        if img.ndim == 2:
            raise ValueError(f"Image {path} is single-channel; expected 4 channels (RGB+IR).")
        if img.shape[2] == 4:
            return img.copy()
        elif img.shape[2] > 4:
            return img[:, :, :4].copy()
        else:
            raise ValueError(f"Image {path} has {img.shape[2]} channels; expected 4 (RGB+IR).")

    def _read_mask(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, "labels", f"{name}.png")
        mask = np.asarray(Image.open(path))
      
        if mask.dtype != np.uint8 and mask.dtype != np.int32 and mask.dtype != np.int64:
            mask = mask.astype(np.int32)
        return mask.copy()

    # ---------- __getitem__ ----------
    def __getitem__(self, index: int):
        name = self.names[index]

        img4 = self._read_image_4ch(name)           # (H, W, 4)
        rgb  = img4[..., :3].astype(np.uint8)       # (H, W, 3)
        ir   = img4[..., 3:]                        # (H, W, 1) — uint8 

        if self.have_label and self.split in ["train", "val","test","test_day","test_night"]:
            mask = self._read_mask(name)            # (H, W)
        else:
            mask = None

        # 1) Shared geometric transforms
        if mask is not None:
            out = self.shared_geom(image=rgb, ir=ir, mask=mask)
            rgb, ir, mask = out["image"], out["ir"], out["mask"]
        else:
            out = self.shared_geom(image=rgb, ir=ir)
            rgb, ir = out["image"], out["ir"]

        # 2) RGB+mask 
        rgb = self.resize_rgb_mask(image=rgb)["image"]

        # 3) IR 
        ir = self.resize_ir(image=ir)["image"]

        mask = self.resize_label(image=mask)["image"] if mask is not None else None

        # 4) Normalize + ToTensor
        rgb = self.rgb_only(image=rgb)["image"]    
        ir  = self.ir_only(image=ir)["image"]      

        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.long) 
            return rgb, ir, mask,name
        else:
            return rgb, ir,name

    def __len__(self) -> int:
        return self.n_data
    
if __name__ == '__main__':
    data_dir = 'Dataset/'
    dataset = FusionModelDataset(data_dir=data_dir, split='train', have_label=True)
    dataset.get_train_item(0)
