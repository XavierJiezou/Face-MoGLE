import gradio as gr
from PIL import Image
import torch
import yaml
import numpy as np
from torchvision import transforms as T
from src.flux.generate import generate
from diffusers.pipelines import FluxPipeline
from src.flux.condition import Condition
from src.moe.mogle import MoGLE


class ImageGenerator:
    def __init__(self):
        self.args = self.get_args()
        self.pipeline, self.moe_model = self.get_model(self.args)
        with open(self.args.config_path, "r") as f:
            self.model_config = yaml.safe_load(f)["model"]

    def get_args(self):
        class Args:
            pipe = "flux"
            lora_ckpt = "runs/face-mogle/pytorch_lora_weights.safetensors"
            moe_ckpt = "runs/face-mogle/mogle.pt"
            pretrained_ckpt = "checkpoints/FLUX.1-dev"
            device = "cuda"
            size = 512
            seed = 42
            config_path = "runs/face-mogle/config.yaml"
        return Args()

    def get_model(self, args):
        pipeline = FluxPipeline.from_pretrained(args.pretrained_ckpt, torch_dtype=torch.bfloat16)
        pipeline.load_lora_weights(args.lora_ckpt)
        pipeline.to(args.device)
        moe_model = MoGLE()
        moe_weight = torch.load(args.moe_ckpt, map_location="cpu")
        moe_model.load_state_dict(moe_weight, strict=True)
        moe_model = moe_model.to(device=args.device, dtype=torch.bfloat16)
        moe_model.eval()
        return pipeline, moe_model

    def pack_data(self, mask_image: Image.Image):
        mask = np.array(mask_image.convert("L"))
        mask_list = [T.ToTensor()(mask_image.convert("RGB"))]
        for i in range(19):
            local_mask = np.zeros_like(mask)
            local_mask[mask == i] = 255
            local_mask_tensor = T.ToTensor()(Image.fromarray(local_mask).convert("RGB"))
            mask_list.append(local_mask_tensor)
        condition_img = torch.stack(mask_list, dim=0)
        return Condition(
            condition_type="depth",
            condition=condition_img,
            position_delta=[0, 0],
        )

    def generate(self, prompt: str, mask_image: Image.Image, seed: int):
        generator = torch.Generator().manual_seed(seed)
        condition = self.pack_data(mask_image)
        result = generate(
            self.pipeline,
            mogle=self.moe_model,
            prompt=prompt,
            conditions=[condition],
            height=self.args.size,
            width=self.args.size,
            generator=generator,
            model_config=self.model_config,
            default_lora=True,
        )
        return result.images[0]


generator = ImageGenerator()

def inference(prompt, mask, seed):
    return generator.generate(prompt, mask, seed)


demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Image(type="pil", label="Mask Image"),
        gr.Slider(minimum=0, maximum=100000, step=1, value=42, label="Random Seed"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Mixture of Global and Local Experts with Diffusion Transformer for Controllable Face Generation",
    # description="Controlnet Face Generation.",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
        share=False,
    )

