"""https://github.com/GaParmar/clean-fid
@inproceedings{parmar2021cleanfid,
  title={On Aliased Resizing and Surprising Subtleties in GAN Evaluation},
  author={Parmar, Gaurav and Zhang, Richard and Zhu, Jun-Yan},
  booktitle={CVPR},
  year={2022}
}
"""
from cleanfid import fid # pip install clean-fid==0.1.35
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from pytorch_lightning import seed_everything


def compute_fid(fdir1, fdir2):
    return fid.compute_fid(fdir1, fdir2, batch_size=50, verbose=False)

def compute_kid(fdir1, fdir2):
    return fid.compute_kid(fdir1, fdir2, batch_size=50, verbose=False)*1000

"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could 
measure the similarity of cross modalities. Please find more information from 
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate 
the mean average of cosine similarities. 

See --help to see further details.

Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/openai/CLIP.

Copyright 2023 The Hong Kong Polytechnic University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import urllib
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
    
    
_MODELS = {
    "FaRL-ViT-B/16-ep16": "https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth",
    "FaRL-ViT-B/16-ep64": "https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
}

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--clip_model', type=str, default='ViT-B/16',
                    help='CLIP model to use')
parser.add_argument('--face_clip', type=str, default=None,
                    help='weather to use FaRL as FaceCLIP') # options: FaRL-ViT-B/16-ep16, FaRL-ViT-B/16-ep64 
parser.add_argument('--num_workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--real_flag', type=str, default='img',
                    help=('The modality of real path. '
                          'Default to img'))
parser.add_argument('--fake_flag', type=str, default='txt',
                    help=('The modality of real path. '
                          'Default to txt'))

parser.add_argument("--fake_image",default="visulization/face-mogle/512/face")
parser.add_argument("--real_face_dir",default="visulization/mmcelebahq/face")
parser.add_argument("--real_text_dir",default="visulization/mmcelebahq/text")
parser.add_argument("--output_dir",default="eval_result")
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

TEXT_EXTENSIONS = {'txt'}


class DummyDataset(Dataset):
    
    FLAGS = ['img', 'txt']
    def __init__(self, real_path, fake_path,
                 real_flag: str = 'img',
                 fake_flag: str = 'img',
                 transform = None,
                 tokenizer = None) -> None:
        super().__init__()
        assert real_flag in self.FLAGS and fake_flag in self.FLAGS, \
            'CLIP Score only support modality of {}. However, get {} and {}'.format(
                self.FLAGS, real_flag, fake_flag
            )
        self.real_folder = self._combine_without_prefix(real_path)
        self.real_flag = real_flag
        self.fake_foler = self._combine_without_prefix(fake_path)
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer
        # assert self._check()

    def __len__(self):
        return len(self.real_folder)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_path = self.real_folder[index]
        fake_path = self.fake_foler[index]
        real_data = self._load_modality(real_path, self.real_flag)
        fake_data = self._load_modality(fake_path, self.fake_flag)

        sample = dict(real=real_data, fake=fake_data)
        return sample
    
    def _load_modality(self, path, modality):
        if modality == 'img':
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        else:
            raise TypeError("Got unexpected modality: {}".format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def _load_txt(self, path):
        with open(path, 'r') as fp:
            data = fp.readline()
            fp.close()
        if self.tokenizer is not None:
            data = self.tokenizer(data).squeeze()
        return data

    def _check(self):
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split('.')
            fake_name = self.fake_folder[idx].split('.')
            if fake_name != real_name:
                return False
        return True
    
    def _combine_without_prefix(self, folder_path, prefix='.'):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder


@torch.no_grad()
def calculate_clip_score(dataloader, model, real_flag, fake_flag):
    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()
    for batch_data in tqdm(dataloader):
        real = batch_data['real']
        real_features = forward_modality(model, real, real_flag)
        fake = batch_data['fake']
        fake_features = forward_modality(model, fake, fake_flag)
        
        # normalize features
        real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)
        
        # calculate scores
        # score = logit_scale * real_features @ fake_features.t()
        # score_acc += torch.diag(score).sum()
        score = logit_scale * (fake_features * real_features).sum()
        score_acc += score
        sample_num += real.shape[0]
    
    return score_acc / sample_num

        
def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def compute_text(fdir1, fdir2):
    image_path = fdir1
    text_path = fdir2
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    print('Loading CLIP model: {}'.format(args.clip_model))
    device = "cuda"
    model, preprocess = clip.load(args.clip_model, device=device, download_root="checkpoints")
    
    if args.face_clip is not None:
        model, preprocess = clip.load("ViT-B/16", device="cpu", download_root="checkpoints")
        if args.face_clip in _MODELS:
            model_path = _download(_MODELS[args.face_clip], root=os.path.expanduser("~/.cache/clip"))
            model = model.to(device)
            farl_state=torch.load(model_path) # you can download from https://github.com/FacePerceiver/FaRL#pre-trained-backbones
            model.load_state_dict(farl_state["state_dict"],strict=False)
        else:
            raise ValueError(f"Unknown FaceCLIP model {args.face_clip}")
        
    
    dataset = DummyDataset(image_path, text_path,
                           args.real_flag, args.fake_flag,
                           transform=preprocess, tokenizer=clip.tokenize)
    dataloader = DataLoader(dataset, args.batch_size, 
                            num_workers=num_workers, pin_memory=True)
    
    # print('Calculating CLIP Score:')
    clip_score = calculate_clip_score(dataloader, model,
                                      args.real_flag, args.fake_flag)
    clip_score = clip_score.cpu().item()
    print(clip_score)
    return clip_score


import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from rich.progress import track
import torch

class MaskDataset(Dataset):
    def __init__(self, dir1, dir2):
        self.dir1 = dir1
        self.dir2 = dir2
        self.files1 = os.listdir(dir1)
        self.files2 = os.listdir(dir2)
        assert len(self.files1) == len(self.files2), "The number of files in both directories should be the same."

    def __len__(self):
        return len(self.files1)

    def __getitem__(self, idx):
        file1 = self.files1[idx]
        file2 = self.files2[idx]
        mask1 = Image.open(os.path.join(self.dir1, file1))
        mask2 = Image.open(os.path.join(self.dir2, file2))
        mask1 = np.array(mask1).astype(np.uint8)
        mask2 = np.array(mask2).astype(np.uint8)
        return mask1, mask2

def compute_mask(fdir1, fdir2, batch_size=50):
    dataset = MaskDataset(fdir1, fdir2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    mask_accuracies = []
    mask_ious = []
    
    for batch in track(dataloader, total=len(dataloader)):
        masks1, masks2 = batch
        masks1 = torch.tensor(masks1)
        masks2 = torch.tensor(masks2)
        
        batch_accuracies = (masks1 == masks2).float().mean(dim=(1, 2)).numpy()
        mask_accuracies.extend(batch_accuracies)
        
        num_classes = int(torch.max(torch.cat((masks1, masks2))) + 1)
        ious = []
        for c in range(num_classes):
            pred_c = (masks2 == c)
            gt_c = (masks1 == c)
            intersection = (pred_c & gt_c).float().sum(dim=(1, 2))
            union = (pred_c | gt_c).float().sum(dim=(1, 2))

            iou_c = torch.where(union == 0, torch.ones_like(intersection), intersection / union)
            ious.append(iou_c)

        ious = torch.stack(ious, dim=1)  # shape: (batch_size, num_classes)
        batch_miou = ious.mean(dim=1).numpy()  
        mask_ious.extend(batch_miou)
    
    avg_mask_accuracy = np.mean(mask_accuracies)
    avg_mask_iou = np.mean(mask_ious)
    
    return float(avg_mask_accuracy * 100), float(avg_mask_iou * 100)


class MultiModalFaceGenerationEvaluator:
    
    seed_everything(42)
    
    def __new__(
            cls,
            real_face_dir,
            fake_face_dir,
            real_text_dir,
        ):
        
        
        results = {}
        results["FID"] = compute_fid(real_face_dir, fake_face_dir)
        results["KID"] = compute_kid(real_face_dir, fake_face_dir)

        results["Text"] = compute_text(fake_face_dir, real_text_dir)

        

        formatted_results = {}
        for key, value in results.items():
            if isinstance(value, (int, float)):
                formatted_results[key] = f"{value:.2f}"
            else:
                formatted_results[key] = value

        evaluation_metrics= {
            "Evaluation Metrics": {
                "FID": str(formatted_results["FID"]),
                "KID": str(formatted_results["KID"]),
                "Text (%)": str(formatted_results["Text"]),
            },
        }
        output_dir = parser.parse_args().output_dir
        os.makedirs(output_dir, exist_ok=True)

        name = parser.parse_args().fake_image.split(os.path.sep)[-3]
        save_path = f"{output_dir}/eval_{name}.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import json

        with open(save_path, "w") as f:
            json.dump(evaluation_metrics, f, indent=4)
        
        print(f"Evaluation metrics saved to {save_path}")

        console = Console()
        table = Table(title="Evaluation Metrics")


        for key in evaluation_metrics["Evaluation Metrics"].keys():
            table.add_column(key, justify="center", style="cyan", no_wrap=True)


        table.add_row(*evaluation_metrics["Evaluation Metrics"].values())

        # os.system('clear')
        console.print(table)

        return evaluation_metrics
    
if __name__ == "__main__":
    
    # TediGAN
    MultiModalFaceGenerationEvaluator(
        real_face_dir=parser.parse_args().real_face_dir,
        fake_face_dir=parser.parse_args().fake_image,
        real_text_dir=parser.parse_args().real_text_dir,
    )
