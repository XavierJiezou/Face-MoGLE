from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import torch
import json


class MMCelebAHQ(Dataset):
    def __init__(
        self,
        root="data/mmcelebahq",
        condition_size: int = 512,
        target_size: int = 512,
        condition_type: str = "depth",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.root = root
        self.face_paths, self.mask_paths, self.prompts = self.get_face_mask_prompt()
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

    def get_face_mask_prompt(self):
        face_paths = [
            os.path.join(self.root, "face", f"{i}.jpg") for i in range(0, 27000)
        ]
        mask_paths = [
            os.path.join(self.root, "mask", f"{i}.png") for i in range(0, 27000)
        ]
        with open(os.path.join(self.root, "text.json"), mode="r") as f:
            prompts = json.load(f)
        return face_paths, mask_paths, prompts

    def __len__(self):
        return len(self.face_paths)

    def __getitem__(self, idx):
        image = Image.open(self.face_paths[idx]).convert("RGB")
        prompts = self.prompts[f"{idx}.jpg"]
        description = random.choices(prompts, k=1)[0].strip()
        enable_scale = random.random() < 1
        if not enable_scale:
            condition_size = int(self.condition_size * self.position_scale)
            position_scale = 1.0
        else:
            condition_size = self.condition_size
            position_scale = self.position_scale

        # Get the condition image
        position_delta = np.array([0, 0])

        mask = np.array(Image.open(self.mask_paths[idx]))
        mask_list = [self.to_tensor(Image.open(self.mask_paths[idx]).convert("RGB"))]
        for i in range(19):
            local_mask = np.zeros_like(mask)
            local_mask[mask == i] = 255

            drop_image = random.random() < self.drop_image_prob
            if drop_image:
                local_mask = np.zeros_like(mask)

            local_mask_rgb = Image.fromarray(local_mask).convert("RGB")
            local_mask_tensor = self.to_tensor(local_mask_rgb)
            mask_list.append(local_mask_tensor)
        condition_img = torch.stack(mask_list,dim=0)


        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        # drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""

        return {
            "image": self.to_tensor(image),
            "condition": condition_img,
            # "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            "position_delta": position_delta,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale": position_scale} if position_scale != 1.0 else {}),
        }

 

