import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
from peft import LoraConfig, get_peft_model_state_dict

import prodigyopt
import os
from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, prepare_text_input

from ..moe.mogle import MoGLE


class FaceMoGLE(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        has_expert=True,
        has_gating=True,
        weight_is_scale=False
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = (
            FluxPipeline.from_pretrained(flux_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        self.mogle = MoGLE(has_expert=has_expert,has_gating=has_gating,weight_is_scale=weight_is_scale)
        self.mogle.train()
        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        torch.save(self.mogle.state_dict(), os.path.join(path, "mogle.pt"))


    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers + [p for p in self.mogle.parameters()]

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        conditions = batch["condition"] # bsx20x3x512x512
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]
        position_scale = float(batch.get("position_scale", [1.0])[0])

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)

            # Prepare text input
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_pipe, prompts
            )

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

        # Prepare conditions # condition_latents \in bsx64x32x32 -> bsx(32x32)x64, condition_ids \in [1024, 3]
        # intial conditions shape [bs, 19, 3, 512, 512] reshape to [bsx19, 3, 512, 512]
        c_bs, c_classes, c_channels, c_h, c_w = conditions.shape
        conditions = conditions.view(c_bs * c_classes, c_channels, c_h, c_w)

        condition_latents, condition_ids = encode_images(self.flux_pipe, conditions) 
        condition_latents_reshape = condition_latents.reshape(c_bs, c_classes, *condition_latents.shape[-2:]) # bs 20 1024 64
        condition_latents = self.mogle.forward(condition_latents_reshape,noise_latent=x_t,timestep=t)
        # conditions shape [bsx19, 1024, 64] # this is condition features
        # condition_ids shape [1024, 3] # this is position embedding
        # help me design a simple MoE to fuse 19 condition_latents
    

        # Add position delta
        condition_ids[:, 1] += position_delta[0]
        condition_ids[:, 2] += position_delta[1]

        if position_scale != 1.0:
            scale_bias = (position_scale - 1.0) / 2
            condition_ids[:, 1] *= position_scale
            condition_ids[:, 2] *= position_scale
            condition_ids[:, 1] += scale_bias
            condition_ids[:, 2] += scale_bias

        # Prepare condition type
        condition_type_ids = torch.tensor(
            [
                Condition.get_type_id(condition_type)
                for condition_type in condition_types
            ]
        ).to(self.device)
        condition_type_ids = (
            torch.ones_like(condition_ids[:, 0]) * condition_type_ids[0]
        ).unsqueeze(1)

        # Prepare guidance
        guidance = (
            torch.ones_like(t).to(self.device)
            if self.transformer.config.guidance_embeds
            else None
        )
        # Forward pass
        transformer_out = tranformer_forward(
            self.transformer,
            # Model config
            model_config=self.model_config,
            # Inputs of the condition (new feature)
            condition_latents=condition_latents,
            condition_ids=condition_ids,
            condition_type_ids=condition_type_ids,
            # Inputs to the original transformer
            hidden_states=x_t,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]
        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss
