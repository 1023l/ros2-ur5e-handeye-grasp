"""Action Chunking Transformer (ACT) with CVAE — self-contained PyTorch.

Reference: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
Low-Cost Hardware" (ACT / ALOHA). This is NOT an API wrapper: encoder,
decoder, KL, chunking, and residual head are implemented here.

Optional residual mode: predict delta over geometric prior action chunk so
the network focuses on dock/perception corrections rather than relearning
the whole pick-place trajectory.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from arm_system.policy.constants import ACTION_DIM, CHUNK_SIZE, IMG_H, IMG_W, STATE_DIM


class ConvEncoder(nn.Module):
    """Lightweight CNN (no torchvision dependency) → token sequence."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) in [0,1]
        feat = self.net(x)  # (B,C,h,w)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, h*w, C)
        return tokens


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = x if mem is None else mem
        h, _ = self.attn(self.ln1(x), self.ln1(ctx), self.ln1(ctx), need_weights=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class ACTPolicy(nn.Module):
    """
    Inputs:
      image (B,3,H,W), state (B,state_dim)
      optional geom_prior_chunk (B,chunk,action_dim) for residual mode
    Train targets:
      action_chunk (B,chunk,action_dim)
    """

    def __init__(
        self,
        *,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = CHUNK_SIZE,
        hidden: int = 256,
        nhead: int = 4,
        n_enc: int = 2,
        n_dec: int = 3,
        latent_dim: int = 32,
        residual: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.residual = residual
        self.hidden = hidden

        self.backbone = ConvEncoder(out_dim=128)
        self.img_proj = nn.Linear(128, hidden)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.pos_img = nn.Parameter(torch.randn(1, 128, hidden) * 0.02)  # covers 120x160 CNN grid
        self.cls = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

        self.enc_blocks = nn.ModuleList(
            [TransformerBlock(hidden, nhead, dropout) for _ in range(n_enc)]
        )

        # CVAE encoder: q(z | obs, actions)
        self.vae_action_proj = nn.Linear(chunk_size * action_dim, hidden)
        self.vae_mu = nn.Linear(hidden * 2, latent_dim)
        self.vae_logvar = nn.Linear(hidden * 2, latent_dim)

        # Decoder queries
        self.query = nn.Parameter(torch.randn(1, chunk_size, hidden) * 0.02)
        self.z_proj = nn.Linear(latent_dim, hidden)
        self.dec_blocks = nn.ModuleList(
            [TransformerBlock(hidden, nhead, dropout) for _ in range(n_dec)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        # Explicit AGV late-dock XY offset from vision (meters, raw — not z-scored)
        self.dock_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 2),
        )
        if residual:
            self.prior_proj = nn.Linear(action_dim, hidden)
        else:
            self.prior_proj = None

    def _encode_obs(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        tokens = self.img_proj(self.backbone(image))  # (B,N,H)
        n = tokens.shape[1]
        tokens = tokens + self.pos_img[:, :n, :]
        st = self.state_proj(state).unsqueeze(1)
        cls = self.cls.expand(image.shape[0], -1, -1)
        x = torch.cat([cls, st, tokens], dim=1)
        for blk in self.enc_blocks:
            x = blk(x)
        return x  # (B, 1+1+N, H)

    def _sample_z(
        self, obs_mem: torch.Tensor, action_chunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = action_chunk.shape[0]
        flat = action_chunk.reshape(b, -1)
        a_h = self.vae_action_proj(flat)
        obs_h = obs_mem[:, 0, :]  # CLS
        h = torch.cat([obs_h, a_h], dim=-1)
        mu = self.vae_mu(h)
        logvar = self.vae_logvar(h).clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def _decode(
        self,
        obs_mem: torch.Tensor,
        z: torch.Tensor,
        geom_prior: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b = obs_mem.shape[0]
        q = self.query.expand(b, -1, -1) + self.z_proj(z).unsqueeze(1)
        if self.residual and geom_prior is not None and self.prior_proj is not None:
            q = q + self.prior_proj(geom_prior)
        for blk in self.dec_blocks:
            q = blk(q, mem=obs_mem)
        out = self.head(q)
        if self.residual and geom_prior is not None:
            out = out + geom_prior
        return out

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        action_chunk: Optional[torch.Tensor] = None,
        geom_prior: Optional[torch.Tensor] = None,
        *,
        kl_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        obs_mem = self._encode_obs(image, state)
        if action_chunk is not None:
            z, mu, logvar = self._sample_z(obs_mem, action_chunk)
        else:
            # inference: prior z=0
            z = torch.zeros(image.shape[0], self.latent_dim, device=image.device)
            mu = z
            logvar = z
        pred = self._decode(obs_mem, z, geom_prior)
        dock = self.dock_head(obs_mem[:, 0, :])
        out: Dict[str, torch.Tensor] = {
            "actions": pred,
            "dock_offset": dock,
            "mu": mu,
            "logvar": logvar,
        }
        if action_chunk is not None:
            l1 = F.l1_loss(pred, action_chunk)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            out["l1"] = l1
            out["kl"] = kl
            out["loss"] = l1 + kl_weight * kl
        return out

    @torch.no_grad()
    def act(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        geom_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        out = self.forward(image, state, action_chunk=None, geom_prior=geom_prior)
        return out["actions"]

    @torch.no_grad()
    def predict_dock_offset(
        self, image: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        obs = self._encode_obs(image, state)
        return self.dock_head(obs[:, 0, :])


def kl_anneal(step: int, warmup: int = 2000) -> float:
    if warmup <= 0:
        return 1.0
    return float(min(1.0, step / warmup))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
