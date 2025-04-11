import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
import torch.optim as optim
from torch.nn import functional as F


# Define the Expert Network
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_softmax=False):
        super(Expert, self).__init__()

        self.use_softmax = use_softmax

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return (
            self.net(x) if not self.use_softmax else torch.softmax(self.net(x), dim=1)
        )


class DynamicGatingNetwork(nn.Module):
    def __init__(self, hidden_dim=64, embed_dim=64, dtype=torch.bfloat16):
        super().__init__()

        self.time_proj = Timesteps(
            hidden_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedding = TimestepEmbedding(hidden_dim, embed_dim)
        self.timestep_embedding = self.timestep_embedding.to(dtype=torch.bfloat16)

        self.noise_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dtype = dtype


        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20), 
        )

    def forward(self, condition_latents, noise_latent, timestep):
        """
        global_latents: (bs, 1024, 64)
        noise_latent: (bs, 1024, 64)
        timestep: (bs,)
        """
        bs, seq_len, hidden_dim = condition_latents.shape


        time_emb = self.time_proj(timestep)  # (bs, hidden_dim)
        time_emb = time_emb.to(self.dtype)
        time_emb = self.timestep_embedding(time_emb)  # (bs, embed_dim)

        time_emb = time_emb.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # (bs, 1024, embed_dim)


        noise_emb = self.noise_proj(noise_latent)  # (bs, 1024, 64)

        # fused_input = torch.cat([condition_latents, noise_emb, time_emb], dim=2)  # (bs, 1024, 64+64+128)
        fused_input = condition_latents + noise_emb + time_emb

        weight = self.gate(fused_input)  # (bs, 1024, 2)
        weight = F.softmax(weight, dim=2)  

        return weight

class MoGLE(nn.Module):
    def __init__(
        self,
        num_experts=20,
        input_dim=64,
        hidden_dim=32,
        output_dim=64,
        has_expert=True,
        has_gating=True,
        weight_is_scale=False,
    ):
        super().__init__()
        expert_model = None
        if has_expert:
            expert_model = Expert
        else:
            expert_model = nn.Identity
        self.global_expert = expert_model(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )
        self.local_experts = nn.ModuleList(
            [
                expert_model(
                    input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
                )
                for _ in range(num_experts - 1)
            ]
        )
        # self.gating = Gating(input_dim=input_dim, num_experts=num_experts)
        if has_gating:
            self.gating = DynamicGatingNetwork()
        else:
            self.gating = nn.Identity()

        self.weight_is_scale = weight_is_scale

    def forward(self, x: torch.Tensor, noise_latent, timestep):
        global_mask = x[:, 0]  # bs 1024 64
        local_mask = x[:, 1:]  # bs 19 1024 64
        if not isinstance(self.gating, nn.Identity):
            weights = self.gating.forward(
                global_mask, noise_latent=noise_latent, timestep=timestep
            )  # bs 1024 20

        _, num_local, h, w = local_mask.shape
        global_output = self.global_expert(global_mask).unsqueeze(1)
        local_outputs = torch.stack(
            [self.local_experts[i](local_mask[:, i]) for i in range(num_local)], dim=1
        )  # (bs, 19, 1024, 64)
        global_local_outputs = torch.cat(
            [global_output, local_outputs], dim=1
        )  # bs 20 1024 64

        if isinstance(self.gating, nn.Identity):
            global_local_outputs = global_local_outputs.sum(dim=1)
            return global_local_outputs
        if self.weight_is_scale:
            weights = torch.mean(weights, dim=1, keepdim=True)  # bs 1 20
            # print("gating scale")

        weights_expanded = weights.unsqueeze(-1)
        output = (global_local_outputs.permute(0, 2, 1, 3) * weights_expanded).sum(
            dim=2
        )
        return output  # bs 1024 64
