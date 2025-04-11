import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os
from torchvision import transforms as T
try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


class FaceMoGLECallback(L.Callback):

    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./runs")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def to_tensor(self, x):
        return T.ToTensor()(x)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )
            if hasattr(pl_module, "save_moe"):
                pl_module.save_moe(
                    f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}/moe.pt"
                )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                batch["condition_type"][
                    0
                ],  # Use the condition type from the current batch
            )


    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type="super_resolution",
    ):
        # TODO: change this two variables to parameters
        target_size = trainer.training_config["dataset"]["target_size"]
        position_scale = trainer.training_config["dataset"].get("position_scale", 1.0)

        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        test_list = []

        condition_img_path = "data/mmcelebahq/mask/27000.png"

        # condition_img = self.deepth_pipe(condition_img)["depth"].convert("RGB")
        test_list.append(
            (
                condition_img_path,
                [0, 0],
                "She is wearing lipstick. She is attractive and has straight hair.",
                {"position_scale": position_scale} if position_scale != 1.0 else {},
            )
        )
        

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, (condition_img_path, position_delta, prompt, *others) in enumerate(
            test_list
        ):

            global_mask = Image.open(condition_img_path).convert("RGB")
            mask_list = [self.to_tensor(global_mask)]
            mask = Image.open(condition_img_path)
            mask = np.array(mask)
            for i in range(19):
                local_mask = np.zeros_like(mask)
                local_mask[mask == i] = 255

                local_mask_rgb = Image.fromarray(local_mask).convert("RGB")
                local_mask_tensor = self.to_tensor(local_mask_rgb)
                mask_list.append(local_mask_tensor)
            condition_img = torch.stack(mask_list, dim=0)
            # condition_img = condition_img.unsqueeze(0)

            condition = Condition(
                condition_type=condition_type,
                condition=condition_img,
                position_delta=position_delta,
                **(others[0] if others else {}),
            )

            res = generate(
                pl_module.flux_pipe,
                mogle=pl_module.mogle,
                prompt=prompt,
                conditions=[condition],
                height=target_size,
                width=target_size,
                generator=generator,
                model_config=pl_module.model_config,
                default_lora=True,
            )
            res.images[0].save(
                os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
            )
