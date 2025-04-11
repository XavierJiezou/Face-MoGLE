from src.flux.generate import generate
from diffusers.pipelines import FluxPipeline
from src.flux.condition import Condition
from PIL import Image
from torchvision import transforms as T
import torch
import argparse
import yaml
import numpy as np
import json
import os
from glob import glob
from natsort import natsorted
from src.moe.mogle import MoGLE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",default="data/mmcelebahq")
    parser.add_argument("--lora_ckpt", default="runs/face-mogle/pytorch_lora_weights.safetensors")
    parser.add_argument("--moe_ckpt", default="runs/face-mogle/mogle.pt")
    parser.add_argument("--pretrained_ckpt", default="checkpoints/FLUX.1-dev")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", default=512)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--config_path", default="runs/face-mogle/config.yaml")
    parser.add_argument("--output_dir", default="visualization/face-mogle")
    args = parser.parse_args()
    return args

def get_model(args):
    pipeline = FluxPipeline.from_pretrained(args.pretrained_ckpt, torch_dtype=torch.bfloat16)
    pipeline.load_lora_weights(args.lora_ckpt)
    pipeline.to(args.device)
    mogle = MoGLE()
    moe_weight = torch.load(args.moe_ckpt,map_location="cpu")
    mogle.load_state_dict(moe_weight, strict=True)
    mogle = mogle.to(device=args.device, dtype=torch.bfloat16)
    mogle.eval()
    return pipeline, mogle

def save_result(image, output_dir, mask_path):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(mask_path).replace(".png",".jpg")
    image.save(os.path.join(output_dir, base_name))
    print(f"Generated image has been saved in {os.path.join(output_dir, base_name)}")

def to_tensor(x):
    return T.ToTensor()(x)

def pack_data(mask_path):
    global_mask = Image.open(mask_path).convert("RGB")
    mask_list = [to_tensor(global_mask)]
    mask = Image.open(mask_path)
    for i in range(19):
        local_mask = np.zeros_like(mask)
        local_mask[mask == i] = 255

        local_mask_rgb = Image.fromarray(local_mask).convert("RGB")
        local_mask_tensor = to_tensor(local_mask_rgb)
        mask_list.append(local_mask_tensor)
    condition_img = torch.stack(mask_list,dim=0)
    condition = Condition(
        condition_type="depth",
        condition=condition_img,
        position_delta=[0, 0],
    )

    return condition

def prepare_data(args):
    print("prepare data.")
    root = args.root
    mask_paths = natsorted(glob(f"{root}/mask/*.png"))
    image_id = [os.path.basename(i).replace(".png",".jpg") for i in mask_paths]
    
    # Load prompts
    prompt_list = []
    with open(os.path.join(root, "text.json"), mode="r") as f:
        prompts = json.load(f)
        for i in image_id:
            prompt_list.append(prompts[i][0].strip())
    
    print("prepare data done.")
    return prompt_list, mask_paths


def main():
    args = get_args()
    pipeline, mogle = get_model(args)
    generator = torch.Generator().manual_seed(args.seed)
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    prompt_list, mask_paths = prepare_data(args)
    os.makedirs(args.output_dir,exist_ok=True)
    for prompt, mask_path in zip(prompt_list, mask_paths):
        
        condition = pack_data(mask_path)

        result = generate(
            pipeline,
            mogle=mogle,
            prompt=prompt,
            conditions=[condition],
            height=args.size,
            width=args.size,
            generator=generator,
            model_config=config["model"],
            default_lora=True,
        )
        
        save_result(result.images[0], output_dir=args.output_dir, mask_path=mask_path)
    

if __name__ == "__main__":
    main()
