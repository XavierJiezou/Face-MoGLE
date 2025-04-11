from typing import List
import os
from rich.progress import track
import argparse
import ImageReward as RM
from natsort import natsorted
from glob import glob

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_path",default="visulization/face-mogle/512/face") # change me
    parse.add_argument("--text_path",default="visulization/mmcelebahq/text")
    parse.add_argument("--device",default="cuda")
    parse.add_argument("--model_path",default="checkpoints/image_reward")
    parse.add_argument("--output_dir",default="eval_result")
    args = parse.parse_args()
    return args

def get_prompts(text_path:str)->List[str]:
    prompts = []
    # prompt_paths = [os.path.join(text_path,f"{i}.txt") for i in range(27000,30000)]
    prompt_paths = natsorted(glob(os.path.join(text_path,"*.txt")))

    for prompt_path in prompt_paths:
        with open(prompt_path,mode='r') as f:
            prompt = f.readline()
            prompts.append(prompt.strip())
    return prompts

def get_images(image_path:str)->List[str]:
    image_filenames = natsorted(glob(os.path.join(image_path,"*.jpg")))

    return image_filenames

def main():
    args = get_args()
    model = RM.load("ImageReward-v1.0",download_root=args.model_path,device=args.device)
    prompts = get_prompts(text_path=args.text_path)
    image_filenames = get_images(args.image_path)
    reward_total = 0
    for prompt,image_filename in track(zip(prompts,image_filenames),total=len(prompts)):
        reward_total += model.score(prompt, [image_filename])
    print(f"avg reward is:{reward_total / len(prompts)}")

    name = args.image_path.split(os.path.sep)[-3]
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/eval_{name}.json"

    import json
    # 读取现有JSON文件（如果存在）
    data = {}
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            data = json.load(f)

    # 更新数据，添加lpips值
    data['ir'] = reward_total / len(prompts)
    # 写回JSON文件
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()