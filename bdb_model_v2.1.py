#!/usr/bin/env python3
"""
BDB 2026 unified pipeline (v2)
- Stage 1: curriculum pretrain with windowed input->future targets (pos/vel/accel supervision)
- Stage 2: full-input -> official output targets training
- Physics/context features: side/role tokens, height/weight bins, relative-position bias
- Direction normalization to 'left' for all plays
- VV (Velocity-Verlet) integrator in (p,q)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# -----------------------------
# Constants / Config
# -----------------------------

SIDE_TO_ID = {"Offense": 0, "Defense": 1}
ROLES = ["Targeted Receiver", "Passer", "Other Route Runner", "Defensive Coverage", "Other"]
ROLE_TO_ID = {r: i for i, r in enumerate(ROLES)}

@dataclass
class Config:
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    seed: int = 1337

    FIELD_X_MAX: float = 120.0
    FIELD_Y_MAX: float = 53.3

    # Model dims
    d_model: int = 256
    nhead: int = 8
    nlayers_temporal: int = 4
    dropout: float = 0.1

    d_side: int = 8
    d_role: int = 8
    accel_mlp_hidden: int = 512

    # Optim
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # Training control
    patience: int = 20
    min_delta: float = 1e-4
    save_every: int = 30
    val_split: float = 0.1

    # Stage-1 curriculum
    stage1_warmup_epochs: int = 5
    stage1_epochs: int = 200
    stage1_min_in: int = 6
    stage1_min_out: int = 2
    stage1_start_out: int = 6
    stage1_out_step: int = 6
    stage1_out_cap: int = 32
    s1_p_loss: float = 1.0
    s1_v_loss: float = 0.2
    s1_a_loss: float = 0.05

    # Stage-2
    stage2_epochs: int = 200

    # Context refresh
    ctx_update_every: int = 5

    # ODE rollout
    dt: float = 0.1

    # Data columns
    base_dyn_cols: Tuple[str, ...] = ("x", "y", "s", "a", "o", "dir")
    static_side_col: str = "player_side"
    static_role_col: str = "player_role"
    static_h_col: str = "player_height"
    static_w_col: str = "player_weight"


# -----------------------------
# Direction utilities
# -----------------------------

def unify_left_direction(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mirror rightward plays so all samples are 'left' oriented (x,y, dir, o, ball_land)."""
    if 'play_direction' not in df.columns:
        return df
    df = df.copy()
    right = df['play_direction'].eq('right')
    # positions
    if 'x' in df.columns:
        df.loc[right, 'x'] = cfg.FIELD_X_MAX - df.loc[right, 'x']
    if 'y' in df.columns:
        df.loc[right, 'y'] = cfg.FIELD_Y_MAX - df.loc[right, 'y']
    # angles in degrees
    for col in ('dir','o'):
        if col in df.columns:
            df.loc[right, col] = (df.loc[right, col] + 180.0) % 360.0
    # ball landing
    if 'ball_land_x' in df.columns:
        df.loc[right, 'ball_land_x'] = cfg.FIELD_X_MAX - df.loc[right, 'ball_land_x']
    if 'ball_land_y' in df.columns:
        df.loc[right, 'ball_land_y'] = cfg.FIELD_Y_MAX - df.loc[right, 'ball_land_y']
    return df


def build_play_direction_map(df_in: pd.DataFrame) -> pd.Series:
    return (
        df_in[['game_id','play_id','play_direction']]
        .drop_duplicates()
        .set_index(['game_id','play_id'])['play_direction']
    )


def apply_direction_to_df(df: pd.DataFrame, dir_map: pd.Series, cfg: Config) -> pd.DataFrame:
    if 'play_direction' not in df.columns:
        dir_df = dir_map.reset_index()  # -> columns: game_id, play_id, play_direction
        df = df.merge(dir_df, on=['game_id','play_id'], how='left', validate='many_to_one')
    return unify_left_direction(df, cfg)


# -----------------------------
# Static feature parsing / tokenization
# -----------------------------

def parse_height_to_inches(h: str | float | int) -> float:
    """Convert '6-1' -> 73 inches; pass-through floats/ints; else NaN."""
    if isinstance(h, (int, float)):
        return float(h)
    if not isinstance(h, str):
        return float('nan')
    h = h.strip()
    if "-" in h:
        ft, inch = h.split("-", 1)
        if ft.isdigit() and inch.isdigit():
            return int(ft) * 12 + int(inch)
    return float('nan')


def canon_role(r: str) -> str:
    if not isinstance(r, str): return "Other"
    r = r.strip()
    if r in ROLE_TO_ID: return r
    if "Targeted" in r: return "Targeted Receiver"
    if "Passer" in r: return "Passer"
    if "Route" in r: return "Other Route Runner"
    if "Coverage" in r: return "Defensive Coverage"
    return "Other"


# -----------------------------
# Dynamics helpers
# -----------------------------

def compute_vel_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute vx, vy from speed and direction: vx = s*cos(dir_rad), vy = s*sin(dir_rad)."""
    df = df.copy()
    s = pd.to_numeric(df.get('s'), errors='coerce')
    dir_deg = pd.to_numeric(df.get('dir'), errors='coerce')
    dir_rad = np.deg2rad(dir_deg)
    df['vx'] = (s * np.cos(dir_rad)).astype(float).fillna(0.0)
    df['vy'] = (s * np.sin(dir_rad)).astype(float).fillna(0.0)
    return df


# -----------------------------
# Data containers
# -----------------------------

from dataclasses import dataclass

@dataclass
class PlayTensors:
    game_id: int
    play_id: int
    nfl_ids: np.ndarray                 # [N]
    p_in: torch.Tensor                  # [N, T_full, 2]
    q_in: torch.Tensor                  # [N, T_full, 2]
    dyn_in: torch.Tensor                # [N, T_full, 4]  (s,a, o, dir)
    height_w: torch.Tensor              # [N, 2]
    side_id: torch.Tensor               # [N]
    role_id: torch.Tensor               # [N]
    p_tgt: torch.Tensor                 # [Nt, T_out, 2] (stage2 ground truth or stage1 sampled); empty for stage1-build
    tgt_nfl_ids: torch.Tensor           # [Nt]
    frame_ids_out: torch.Tensor         # [T_out]

    def to(self, device: str) -> "PlayTensors":
        return PlayTensors(
            self.game_id, self.play_id, self.nfl_ids,
            self.p_in.to(device), self.q_in.to(device), self.dyn_in.to(device),
            self.height_w.to(device), self.side_id.to(device), self.role_id.to(device),
            self.p_tgt.to(device), self.tgt_nfl_ids.to(device), self.frame_ids_out.to(device)
        )


# -----------------------------
# Build plays (Stage 1 full sequences)
# -----------------------------

def build_stage1_plays(df_in_unified: pd.DataFrame, cfg: Config) -> List[PlayTensors]:
    plays: List[PlayTensors] = []

    df = df_in_unified.copy()
    if "height_in" not in df.columns:
        df["height_in"] = df[cfg.static_h_col].map(parse_height_to_inches)
    if "role_id" not in df.columns:
        df["role_canon"] = df[cfg.static_role_col].map(canon_role)
        df["role_id"] = df["role_canon"].map(lambda r: ROLE_TO_ID[r]).astype(int)
    if "side_id" not in df.columns:
        df["side_id"] = df[cfg.static_side_col].map(lambda s: SIDE_TO_ID.get(s, 0)).astype(int)
    if "weight_lb" not in df.columns:
        df["weight_lb"] = pd.to_numeric(df[cfg.static_w_col], errors="coerce")

    df = compute_vel_xy(df)

    for (g, pid), gdf in df.groupby(["game_id","play_id"]):
        gdf = gdf.sort_values(["nfl_id","frame_id"])
        nfls = gdf["nfl_id"].unique()
        N = len(nfls)
        T_full = int(gdf["frame_id"].max())
        if T_full < 4:
            continue

        P_full = np.zeros((N, T_full, 2), dtype=np.float32)
        Q_full = np.zeros((N, T_full, 2), dtype=np.float32)
        D_full = np.zeros((N, T_full, 4), dtype=np.float32)  # s,a,o,dir
        HW     = np.zeros((N, 2), dtype=np.float32)
        SIDE   = np.zeros((N,), dtype=np.int64)
        ROLE   = np.zeros((N,), dtype=np.int64)

        for i, nid in enumerate(nfls):
            pdf = gdf[gdf["nfl_id"]==nid].sort_values("frame_id")
            pdf = pdf.set_index("frame_id").reindex(range(1, T_full+1)).ffill().reset_index()

            P_full[i, :, 0] = pdf["x"].to_numpy(dtype=np.float32)
            P_full[i, :, 1] = pdf["y"].to_numpy(dtype=np.float32)
            Q_full[i, :, 0] = pdf["vx"].to_numpy(dtype=np.float32)
            Q_full[i, :, 1] = pdf["vy"].to_numpy(dtype=np.float32)
            D_full[i, :, 0] = pdf["s"].to_numpy(dtype=np.float32)
            D_full[i, :, 1] = pdf["a"].to_numpy(dtype=np.float32)
            D_full[i, :, 2] = pdf["o"].to_numpy(dtype=np.float32)
            D_full[i, :, 3] = pdf["dir"].to_numpy(dtype=np.float32)

            HW[i, 0] = float(pdf["height_in"].iloc[0]) if "height_in" in pdf else np.nan
            HW[i, 1] = float(pdf["weight_lb"].iloc[0]) if "weight_lb" in pdf else np.nan
            SIDE[i]  = int(pdf["side_id"].iloc[0]) if "side_id" in pdf else 0
            ROLE[i]  = int(pdf["role_id"].iloc[0]) if "role_id" in pdf else ROLE_TO_ID["Other"]

        plays.append(PlayTensors(
            game_id=int(g), play_id=int(pid), nfl_ids=nfls.astype(np.int64),
            p_in=torch.from_numpy(P_full), q_in=torch.from_numpy(Q_full),
            dyn_in=torch.from_numpy(D_full), height_w=torch.from_numpy(HW),
            side_id=torch.from_numpy(SIDE), role_id=torch.from_numpy(ROLE),
            p_tgt=torch.empty((N, 0, 2), dtype=torch.float32),
            tgt_nfl_ids=torch.from_numpy(nfls.astype(np.int64)),
            frame_ids_out=torch.arange(0, 0, dtype=torch.long)
        ))
    return plays


# -----------------------------
# Build plays (Stage 2 from input + output CSVs)
# -----------------------------

def build_stage2_plays(df_in_unified: pd.DataFrame, df_out_unified: pd.DataFrame, cfg: Config) -> List[PlayTensors]:
    """
    For each (game_id, play_id), build full input (as stage1) and attach targets from output CSV.
    Targets: positions only (x,y) at future frames; we align on nfl_id intersection.
    """
    plays: List[PlayTensors] = []

    # Prepare input same as stage1
    df_in = df_in_unified.copy()
    if "height_in" not in df_in.columns:
        df_in["height_in"] = df_in[cfg.static_h_col].map(parse_height_to_inches)
    if "role_id" not in df_in.columns:
        df_in["role_canon"] = df_in[cfg.static_role_col].map(canon_role)
        df_in["role_id"] = df_in["role_canon"].map(lambda r: ROLE_TO_ID[r]).astype(int)
    if "side_id" not in df_in.columns:
        df_in["side_id"] = df_in[cfg.static_side_col].map(lambda s: SIDE_TO_ID.get(s, 0)).astype(int)
    if "weight_lb" not in df_in.columns:
        df_in["weight_lb"] = pd.to_numeric(df_in[cfg.static_w_col], errors="coerce")
    df_in = compute_vel_xy(df_in)

    # Ensure output has x,y and frame_id; unify left has already mirrored positions
    df_out = df_out_unified.copy()

    # Group on plays present in both
    keys_in = set(df_in.groupby(["game_id","play_id"]).groups.keys())
    keys_out = set(df_out.groupby(["game_id","play_id"]).groups.keys())
    keys = sorted(keys_in & keys_out)

    for (g, pid) in keys:
        gdf_in = df_in[(df_in.game_id==g) & (df_in.play_id==pid)].sort_values(["nfl_id","frame_id"])
        gdf_out = df_out[(df_out.game_id==g) & (df_out.play_id==pid)].sort_values(["nfl_id","frame_id"])

        nfls_in = gdf_in["nfl_id"].unique()
        nfls_out = gdf_out["nfl_id"].unique()
        nfls = np.intersect1d(nfls_in, nfls_out)
        if len(nfls) == 0:
            continue

        T_full = int(gdf_in["frame_id"].max())
        frames_out = np.sort(gdf_out["frame_id"].unique())
        T_out = len(frames_out)
        if T_full < 1 or T_out < 1:
            continue

        N = len(nfls)
        P_full = np.zeros((N, T_full, 2), dtype=np.float32)
        Q_full = np.zeros((N, T_full, 2), dtype=np.float32)
        D_full = np.zeros((N, T_full, 4), dtype=np.float32)  # s,a,o,dir
        HW     = np.zeros((N, 2), dtype=np.float32)
        SIDE   = np.zeros((N,), dtype=np.int64)
        ROLE   = np.zeros((N,), dtype=np.int64)

        P_tgt  = np.zeros((N, T_out, 2), dtype=np.float32)
        tgt_ids= np.zeros((N,), dtype=np.int64)

        # Fill inputs per player
        for i, nid in enumerate(nfls):
            pdf = gdf_in[gdf_in["nfl_id"]==nid].sort_values("frame_id")
            pdf = pdf.set_index("frame_id").reindex(range(1, T_full+1)).ffill().reset_index()
            P_full[i, :, 0] = pdf["x"].to_numpy(dtype=np.float32)
            P_full[i, :, 1] = pdf["y"].to_numpy(dtype=np.float32)
            Q_full[i, :, 0] = pdf["vx"].to_numpy(dtype=np.float32)
            Q_full[i, :, 1] = pdf["vy"].to_numpy(dtype=np.float32)
            D_full[i, :, 0] = pdf["s"].to_numpy(dtype=np.float32)
            D_full[i, :, 1] = pdf["a"].to_numpy(dtype=np.float32)
            D_full[i, :, 2] = pdf["o"].to_numpy(dtype=np.float32)
            D_full[i, :, 3] = pdf["dir"].to_numpy(dtype=np.float32)

            HW[i, 0] = float(pdf["height_in"].iloc[0]) if "height_in" in pdf else np.nan
            HW[i, 1] = float(pdf["weight_lb"].iloc[0]) if "weight_lb" in pdf else np.nan
            SIDE[i]  = int(pdf["side_id"].iloc[0]) if "side_id" in pdf else 0
            ROLE[i]  = int(pdf["role_id"].iloc[0]) if "role_id" in pdf else ROLE_TO_ID["Other"]

            # Targets from output
            qdf = gdf_out[(gdf_out["nfl_id"]==nid) & (gdf_out["frame_id"].isin(frames_out))]
            # align by frames_out order
            m = qdf.set_index("frame_id").reindex(frames_out).ffill()
            P_tgt[i, :, 0] = m["x"].to_numpy(dtype=np.float32)
            P_tgt[i, :, 1] = m["y"].to_numpy(dtype=np.float32)
            tgt_ids[i] = int(nid)

        plays.append(PlayTensors(
            game_id=int(g), play_id=int(pid), nfl_ids=nfls.astype(np.int64),
            p_in=torch.from_numpy(P_full), q_in=torch.from_numpy(Q_full),
            dyn_in=torch.from_numpy(D_full), height_w=torch.from_numpy(HW),
            side_id=torch.from_numpy(SIDE), role_id=torch.from_numpy(ROLE),
            p_tgt=torch.from_numpy(P_tgt),
            tgt_nfl_ids=torch.from_numpy(tgt_ids.astype(np.int64)),
            frame_ids_out=torch.from_numpy(frames_out.astype(np.int64))
        ))
    return plays


# -----------------------------
# Stage 1 dynamic dataset (sampling windows)
# -----------------------------

class Stage1DynamicDataset(Dataset):
    """Each __getitem__ samples a (T_in, T_out_eff) slice with T_out_eff ≤ T_out_curr and returns PlayTensors + V/A targets."""
    def __init__(self, plays: List[PlayTensors], cfg: Config, T_out_curr: int):
        self.plays = plays
        self.cfg = cfg
        self.T_out_curr = int(T_out_curr)
        self.rng = np.random.RandomState(cfg.seed)

    def __len__(self) -> int:
        return len(self.plays)

    def __getitem__(self, idx: int):
        pl = self.plays[idx]
        N, T_full = pl.p_in.shape[:2]
        min_in = self.cfg.stage1_min_in

        if T_full <= min_in:
            T_in = max(1, T_full - 1)
            T_out_eff = 1
        else:
            # clamp to what this play can support; ensures T_out_eff ≤ T_out_curr
            T_out_eff = max(1, min(self.T_out_curr, T_full - min_in))
            # randint high is exclusive; choose any Tin in [min_in, T_full - T_out_eff]
            T_in = int(self.rng.randint(min_in, T_full - T_out_eff + 1))

        # slice inputs
        p_in = pl.p_in[:, :T_in, :]
        q_in = pl.q_in[:, :T_in, :]
        d_in = pl.dyn_in[:, :T_in, :]

        # targets for next steps
        p_next = pl.p_in[:, T_in:T_in+T_out_eff, :]
        v_next = pl.q_in[:, T_in:T_in+T_out_eff, :]
        a_full = torch.diff(pl.q_in, dim=1, prepend=pl.q_in[:, :1, :]) / self.cfg.dt
        a_next = a_full[:, T_in:T_in+T_out_eff, :]

        return PlayTensors(
            game_id=pl.game_id,
            play_id=pl.play_id,
            nfl_ids=pl.nfl_ids,
            p_in=p_in,
            q_in=q_in,
            dyn_in=d_in,
            height_w=pl.height_w,
            side_id=pl.side_id,
            role_id=pl.role_id,
            p_tgt=p_next,
            tgt_nfl_ids=torch.from_numpy(pl.nfl_ids.astype(np.int64)),
            frame_ids_out=torch.arange(1, p_next.shape[1]+1)
        ), dict(V_tgt=v_next, A_tgt=a_next)


# -----------------------------
# Simple wrapper for Stage-2 plays as a Dataset
# -----------------------------

class PlaysDataset(Dataset):
    def __init__(self, plays: List[PlayTensors]):
        self.plays = plays
    def __len__(self): return len(self.plays)
    def __getitem__(self, idx): return self.plays[idx]


# -----------------------------
# Collate
# -----------------------------

def collate_plays(batch: List[PlayTensors]) -> Dict[str, torch.Tensor]:
    """Pad to max N and max T for this batch."""
    Tin_list = []
    N_max = max(x.p_in.shape[0] for x in batch)
    T_in_max = max(x.p_in.shape[1] for x in batch)
    T_out_max = max(x.p_tgt.shape[1] for x in batch)

    # allocate
    P_in = torch.zeros(len(batch), N_max, T_in_max, 2)
    Q_in = torch.zeros_like(P_in)
    D_in = torch.zeros(len(batch), N_max, T_in_max, batch[0].dyn_in.shape[-1])
    HW   = torch.zeros(len(batch), N_max, 2)
    SIDE = torch.full((len(batch), N_max), -1, dtype=torch.long)
    ROLE = torch.full((len(batch), N_max), -1, dtype=torch.long)
    NFL  = torch.full((len(batch), N_max), -1, dtype=torch.long)
    mask_in = torch.zeros(len(batch), N_max, dtype=torch.bool)

    # targets
    P_tgt = torch.zeros(len(batch), N_max, T_out_max, 2)
    NFL_t = torch.full((len(batch), N_max), -1, dtype=torch.long)
    mask_t = torch.zeros(len(batch), N_max, dtype=torch.bool)
    F_out  = torch.zeros(len(batch), T_out_max, dtype=torch.long)

    for b, pl in enumerate(batch):
        N, T_in = pl.p_in.shape[:2]
        Tin_list.append(T_in)
        Nt, T_out = pl.p_tgt.shape[:2]

        P_in[b, :N, -T_in:] = pl.p_in
        Q_in[b, :N, -T_in:] = pl.q_in
        D_in[b, :N, -T_in:] = pl.dyn_in

        HW[b, :N] = pl.height_w
        SIDE[b, :N] = pl.side_id
        ROLE[b, :N] = pl.role_id
        NFL[b, :N]  = torch.as_tensor(pl.nfl_ids, dtype=torch.long)
        mask_in[b, :N] = True

        if T_out > 0:
            P_tgt[b, :Nt, -T_out:] = pl.p_tgt
            NFL_t[b, :Nt] = torch.as_tensor(pl.tgt_nfl_ids, dtype=torch.long)
            mask_t[b, :Nt] = True
            F_out[b, -T_out:] = pl.frame_ids_out

    # time key padding mask (True = padded left timesteps)
    time_kpm = torch.zeros(len(batch), T_in_max, dtype=torch.bool)
    for b, Tin in enumerate(Tin_list):
        pad = T_in_max - Tin
        if pad > 0:
            time_kpm[b, :pad] = True

    return dict(
        P_in=P_in, Q_in=Q_in, D_in=D_in, HW=HW, SIDE=SIDE, ROLE=ROLE, NFL=NFL, mask_in=mask_in,
        P_tgt=P_tgt, NFL_t=NFL_t, mask_t=mask_t, F_out=F_out, time_kpm=time_kpm
    )


# -----------------------------
# Model components
# -----------------------------

class TemporalPlayerEncoder(nn.Module):
    """
    Per-player temporal encoder over T_in frames.
    Input per time-step for player i:
    [x, y, s, a, o_sin, o_cos, dir_sin, dir_cos, vx, vy] + statics [h,w, side_emb, role_emb]
    -> project to d_model and pass through TransformerEncoder with key_padding_mask.
    Output: last-token embedding per player, [B,N,d_model].
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.side_emb = nn.Embedding(2, cfg.d_side)
        self.role_emb = nn.Embedding(len(ROLE_TO_ID), cfg.d_role)

        d_dyn = 10  # x,y,s,a,o_sin,o_cos,dir_sin,dir_cos,vx,vy
        d_stat = 2 + cfg.d_side + cfg.d_role
        self.in_proj = nn.Linear(d_dyn + d_stat, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, batch_first=True, dropout=cfg.dropout, dim_feedforward=cfg.d_model*4
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.nlayers_temporal)

    def forward(self, P_in, Q_in, D_in, HW, SIDE, ROLE, mask_in, time_kpm):
        """
        P_in: [B,N,T,2], Q_in: [B,N,T,2], D_in: [B,N,T,4] (s,a,o,dir)
        HW: [B,N,2] raw height/weight; SIDE/ROLE: [B,N] ids; mask_in: [B,N] valid players.
        time_kpm: [B,T] True where padded (left).
        """
        cfg = self.cfg
        B,N,T,_ = P_in.shape
        x = P_in[..., 0]; y = P_in[..., 1]
        s = D_in[..., 0]; a = D_in[..., 1]
        o_deg = D_in[..., 2]; dir_deg = D_in[..., 3]
        o_sin = torch.sin(torch.deg2rad(o_deg)); o_cos = torch.cos(torch.deg2rad(o_deg))
        d_sin = torch.sin(torch.deg2rad(dir_deg)); d_cos = torch.cos(torch.deg2rad(dir_deg))
        vx = Q_in[..., 0]; vy = Q_in[..., 1]
        dyn = torch.stack([x,y,s,a,o_sin,o_cos,d_sin,d_cos,vx,vy], dim=-1)  # [B,N,T,10]

        side_e = self.side_emb(SIDE.clamp_min(0))    # [B,N,d_side]
        role_e = self.role_emb(ROLE.clamp_min(0))    # [B,N,d_role]
        stat = torch.cat([HW, side_e, role_e], dim=-1)  # [B,N,2+d_side+d_role]
        stat = stat.unsqueeze(2).expand(B,N,T,stat.shape[-1])

        x_in = torch.cat([dyn, stat], dim=-1)            # [B,N,T,d_in]
        x_in = self.in_proj(x_in)                        # [B,N,T,d_model]
        x_in = x_in.view(B*N, T, -1)

        # Build key padding mask across time and players
        if time_kpm is None:
            kpm = None
        else:
            assert time_kpm.shape == (B, T)
            kpm_bnT = time_kpm.unsqueeze(1).expand(B, N, T).clone()
            invalid_players = (~mask_in).unsqueeze(-1).expand(B, N, T)
            kpm_bnT |= invalid_players
            kpm = kpm_bnT.view(B*N, T)

        h = self.encoder(x_in, src_key_padding_mask=kpm)    # [B*N,T,d_model]
        h_last = h[:, -1, :].view(B, N, -1)                 # [B,N,d_model]
        h_last = h_last.masked_fill((~mask_in).unsqueeze(-1), 0.0)
        return h_last


class RPBEncoder(nn.Module):
    """
    Relative-position biased interaction encoder.
    Takes per-player embeddings h_i and current positions p_i; adds a learned bias from [dx,dy,dist,is_teammate].
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.nhead = cfg.nhead
        self.bias_mlp = nn.Sequential(
            nn.Linear(4, d), nn.ReLU(), nn.Linear(d, self.nhead)  # per-head bias
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, h, p, side_id, mask):
        """
        h: [B,N,d_model], p: [B,N,2], side_id: [B,N], mask: [B,N]
        returns: [B,N,d_model]
        """

        B,N,D = h.shape
        H = self.nhead; d_head = D // H

        h = h.masked_fill((~mask).unsqueeze(-1), 0.0)

        q = self.q_proj(h).view(B,N,H,d_head).transpose(1,2)  # [B,H,N,d_head]
        k = self.k_proj(h).view(B,N,H,d_head).transpose(1,2)  # [B,H,N,d_head]
        v = self.v_proj(h).view(B,N,H,d_head).transpose(1,2)  # [B,H,N,d_head]

        # Relative positions
        dp = p.unsqueeze(2) - p.unsqueeze(1)  # [B,N,N,2]
        dist = torch.clamp(torch.norm(dp, dim=-1, keepdim=True), min=1e-6)
        is_team = (side_id.unsqueeze(2) == side_id.unsqueeze(1)).float().unsqueeze(-1)  # [B,N,N,1]
        feat = torch.cat([dp, dist, is_team], dim=-1)  # [B,N,N,4]
        bias_per_head = self.bias_mlp(feat)            # [B,N,N,H]
        bias_per_head = bias_per_head.permute(0,3,1,2) # [B,H,N,N]

        attn_logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(d_head)
        attn_logits = attn_logits + bias_per_head

        # mask invalid players
        m = (~mask).float() * (-1e9)
        m = m.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        attn_logits = attn_logits + m

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)       # [B,H,N,d_head]
        out = out.transpose(1,2).contiguous().view(B,N,D)    # [B,N,D]
        out = self.out(out)
        return out


class AccelMLP(nn.Module):
    """Predict per-player acceleration q' = (ax, ay) from context + current velocity + static bins."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.side_emb = nn.Embedding(2, cfg.d_side)
        self.role_emb = nn.Embedding(len(ROLE_TO_ID), cfg.d_role)

        # Height/Weight bins and embeddings
        h_bins = torch.tensor([65, 68, 70, 72, 74, 76, 78, 80], dtype=torch.float32)
        w_bins = torch.tensor([180, 200, 220, 240, 260, 280, 300, 330], dtype=torch.float32)
        self.register_buffer('h_bins', h_bins)
        self.register_buffer('w_bins', w_bins)
        self.h_emb = nn.Embedding(len(h_bins)+1, 8)
        self.w_emb = nn.Embedding(len(w_bins)+1, 8)

        in_dim = (cfg.d_model + cfg.d_model + 2 + cfg.d_side + cfg.d_role) + 8 + 8
        hid = cfg.accel_mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, 2)
        )

    def forward(self, h_temporal, h_rpb, q, hw_std, side_id, role_id):
        height = torch.nan_to_num(hw_std[..., 0], nan=0.0)
        weight = torch.nan_to_num(hw_std[..., 1], nan=0.0)
        h_idx = torch.bucketize(height, self.h_bins)
        w_idx = torch.bucketize(weight, self.w_bins)
        h_e = self.h_emb(h_idx)
        w_e = self.w_emb(w_idx)
        side_e = self.side_emb(side_id.clamp_min(0))
        role_e = self.role_emb(role_id.clamp_min(0))
        x = torch.cat([h_temporal, h_rpb, q, h_e, w_e, side_e, role_e], dim=-1)
        return self.mlp(x)  # [B,N,2]


class NeuralODEModel(nn.Module):
    """Temporal encoder -> RPB -> AccelMLP -> Velocity-Verlet rollout."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.temporal = TemporalPlayerEncoder(cfg)
        self.rpb = RPBEncoder(cfg)
        self.accel = AccelMLP(cfg)

    def rollout(self, h_temporal, p0, q0, hw_std, side_id, role_id, mask, T_out):
        cfg = self.cfg
        B,N,_ = p0.shape
        dt = cfg.dt
        P_list = []
        Q_list = []
        p = p0
        q = q0
        h_rpb_cached = None

        for t in range(T_out):
            if (t % max(1, cfg.ctx_update_every)) == 0 or (h_rpb_cached is None):
                h_rpb_cached = self.rpb(h_temporal, p, side_id, mask)

            # h_rpb1 = self.rpb(h_temporal, p, side_id, mask)

            a1 = self.accel(h_temporal, h_rpb_cached, q, hw_std, side_id, role_id)
            q_half = q + 0.5 * dt * a1
            p_new = p + dt * q_half

            # h_rpb2 = self.rpb(h_temporal, p_new, side_id, mask)

            a2 = self.accel(h_temporal, h_rpb_cached, q_half, hw_std, side_id, role_id)
            q_new = q_half + 0.5 * dt * a2

            p, q = p_new, q_new
            P_list.append(p.unsqueeze(2))
            Q_list.append(q.unsqueeze(2))

        P_pred = torch.cat(P_list, dim=2)
        Q_pred = torch.cat(Q_list, dim=2)
        return P_pred, Q_pred

    def forward(self, batch, stage: str):
        cfg = self.cfg
        P_in = batch["P_in"].to(cfg.device)
        Q_in = batch["Q_in"].to(cfg.device)
        D_in = batch["D_in"].to(cfg.device)
        HW   = batch["HW"].to(cfg.device)
        SIDE = batch["SIDE"].to(cfg.device)
        ROLE = batch["ROLE"].to(cfg.device)
        mask = batch["mask_in"].to(cfg.device)
        P_tgt= batch["P_tgt"].to(cfg.device)   # [B,N_max,T_out,2]
        mask_t = batch["mask_t"].to(cfg.device)
        time_kpm = batch.get("time_kpm", None)
        if time_kpm is not None: time_kpm = time_kpm.to(cfg.device)
        T_out = P_tgt.shape[2] if P_tgt.shape[2] > 0 else 1

        h_temporal = self.temporal(P_in, Q_in, D_in, HW, SIDE, ROLE, mask, time_kpm)  # [B,N,d]
        p0 = P_in[:, :, -1, :]  # last input frame
        q0 = Q_in[:, :, -1, :]
        P_pred, Q_pred = self.rollout(h_temporal, p0, q0, HW, SIDE, ROLE, mask, T_out)
        return P_pred, Q_pred


# -----------------------------
# Loss / metrics
# -----------------------------

def compute_rmse_pos(P_pred, P_tgt, mask_players, time_mask):
    """
    P_pred, P_tgt: [B,N,T,2]
    mask_players: [B,N] (True for real players)
    time_mask:    [B,T] (True for real timesteps; false for left padding)
    """
    B, N, T, C = P_tgt.shape
    mp = mask_players.unsqueeze(-1).unsqueeze(-1)         # [B,N,1,1]
    mt = time_mask.unsqueeze(1).unsqueeze(-1)             # [B,1,T,1]
    m  = (mp & mt).expand(B, N, T, C)                     # [B,N,T,2]

    diff = (P_pred - P_tgt)[m]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=P_pred.device)
    return torch.sqrt(torch.mean(diff**2) + 1e-12)


def compute_rmse_vec(A_pred, A_tgt, mask_players, time_mask):
    B, N, T, C = A_tgt.shape
    mp = mask_players.unsqueeze(-1).unsqueeze(-1)         # [B,N,1,1]
    mt = time_mask.unsqueeze(1).unsqueeze(-1)             # [B,1,T,1]
    m  = (mp & mt).expand(B, N, T, C)
    diff = (A_pred - A_tgt)[m]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=A_pred.device)
    return torch.sqrt(torch.mean(diff**2) + 1e-12)



def compute_stage1_targets(Q_pred, cfg):
    """Return accelerations from predicted velocities via finite difference along time."""
    a = torch.diff(Q_pred, dim=2, prepend=Q_pred[:, :, :1, :]) / cfg.dt
    return a


# -----------------------------
# Data pairing / splits
# -----------------------------

def find_pairs(train_dir: Path) -> List[Tuple[Path, Path]]:
    csvs = [p for p in train_dir.glob("**/*.csv") if p.is_file()]
    ins  = [p for p in csvs if "input"  in p.name.lower()]
    outs = [p for p in csvs if "output" in p.name.lower()]
    if len(ins)==1 and len(outs)==1:
        return [(ins[0], outs[0])]
    def key(name: str) -> Optional[str]:
        name = name.lower()
        if "input"  in name: return name.split("input", 1)[1]
        if "output" in name: return name.split("output", 1)[1]
        return None
    by_in, by_out, pairs = {}, {}, []
    for pth in ins:  by_in.setdefault(key(pth.name), []).append(pth)
    for pth in outs: by_out.setdefault(key(pth.name), []).append(pth)
    for k in sorted(set(by_in) & set(by_out)):
        pairs.append((by_in[k][0], by_out[k][0]))
    if not pairs:
        i, o = train_dir/"train_input.csv", train_dir/"train_output.csv"
        if i.exists() and o.exists(): pairs.append((i, o))
    if not pairs:
        raise FileNotFoundError(f"No input/output CSV pairs found in {train_dir}.")
    return pairs


def train_val_split(plays: List[PlayTensors], val_split: float, seed: int = 1337):
    import random
    idx = list(range(len(plays)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_split))
    val_idx = set(idx[:n_val])
    train = [plays[i] for i in range(len(plays)) if i not in val_idx]
    val = [plays[i] for i in range(len(plays)) if i in val_idx]
    return train, val


# -----------------------------
# Train / Eval
# -----------------------------

def evaluate(model: nn.Module, loader, cfg: Config, stage: str):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            P_pred, Q_pred = model(batch, stage)
            time_mask = (batch["F_out"].to(cfg.device) > 0)
            mask_t = batch["mask_t"].to(cfg.device)
            if stage == "stage2":
                P_tgt = batch["P_tgt"].to(cfg.device)
                loss = compute_rmse_pos(P_pred, P_tgt, mask_t, time_mask)
            else:
                P_tgt = batch["P_tgt"].to(cfg.device)  # still supervise P
                V_tgt = batch["V_tgt"].to(cfg.device)
                A_pred = compute_stage1_targets(Q_pred,cfg)
                A_tgt = batch["A_tgt"].to(cfg.device)
                # Stage-1 loss: position + velocity + acceleration
                loss = (
                    cfg.s1_p_loss*compute_rmse_pos(P_pred, P_tgt, mask_t, time_mask) +
                    cfg.s1_v_loss*compute_rmse_vec(Q_pred, V_tgt, mask_t, time_mask) +
                    cfg.s1_a_loss*compute_rmse_vec(A_pred, A_tgt, mask_t, time_mask)
                )
            losses.append(float(loss))
    return float(np.mean(losses)) if losses else float('inf')


def train(model: nn.Module, loaders: Dict[str, DataLoader], cfg: Config, epochs: int, stage: str, log_prefix=""):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float('inf')
    wait = 0
    hist = {"train": [], "val": []}

    for ep in range(1, epochs+1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            opt.zero_grad()
            P_pred, Q_pred = model(batch, stage)
            mask_t = batch["mask_t"].to(cfg.device)
            time_mask = (batch["F_out"].to(cfg.device) > 0)
            if stage == "stage2":
                P_tgt = batch["P_tgt"].to(cfg.device)
                loss = compute_rmse_pos(P_pred, P_tgt, mask_t, time_mask)
            else:
                P_tgt = batch["P_tgt"].to(cfg.device)
                V_tgt = batch["V_tgt"].to(cfg.device)
                A_pred = compute_stage1_targets(Q_pred,cfg)
                A_tgt = batch["A_tgt"].to(cfg.device)
                loss = (
                    cfg.s1_p_loss*compute_rmse_pos(P_pred, P_tgt, mask_t, time_mask) +
                    cfg.s1_v_loss*compute_rmse_vec(Q_pred, V_tgt, mask_t, time_mask) +
                    cfg.s1_a_loss*compute_rmse_vec(A_pred, A_tgt, mask_t, time_mask)
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            train_losses.append(float(loss))

        val_loss = evaluate(model, loaders["val"], cfg, stage)
        hist["train"].append(float(np.mean(train_losses)))
        hist["val"].append(float(val_loss))
        print(f"[{stage}][epoch {ep}][train {hist['train'][-1]}][val {hist['val'][-1]}]")

        if val_loss + cfg.min_delta < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    return {"best_val": best_val}, hist


# -----------------------------
# Driver
# -----------------------------

def build_all_stage1_plays(pairs: List[Tuple[Path,Path]], cfg: Config) -> List[PlayTensors]:
    plays1 = []
    for pin, _ in pairs:
        df_in_raw = pd.read_csv(pin, engine="c")  # robust to encodings
        dir_map = build_play_direction_map(df_in_raw)
        df_in = apply_direction_to_df(df_in_raw, dir_map, cfg)
        plays1.extend(build_stage1_plays(df_in, cfg))
    return plays1


def build_all_stage2_plays(pairs: List[Tuple[Path,Path]], cfg: Config) -> List[PlayTensors]:
    plays2 = []
    for pin, pout in pairs:
        df_in_raw  = pd.read_csv(pin, engine="c")
        dir_map    = build_play_direction_map(df_in_raw)
        df_in      = apply_direction_to_df(df_in_raw, dir_map, cfg)
        df_out_raw = pd.read_csv(pout, engine="c")
        df_out     = apply_direction_to_df(df_out_raw, dir_map, cfg)
        plays2.extend(build_stage2_plays(df_in, df_out, cfg))
    return plays2


def main():
    import json, datetime
    from dataclasses import asdict

    cfg = Config()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    print(f"Device: {cfg.device}")

    # ----- Run-specific directory -----
    run_root = Path("./runs")
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"run_{run_id}_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] outputs -> {run_dir}")

    # Save config (JSON)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    TRAIN_DIR = Path("./data/raw/train")
    pairs = find_pairs(TRAIN_DIR)
    print("Using pairs:")
    for pin, pout in pairs:
        print("  IN:", pin, " OUT:", pout)

    # Optional: record pairs for reproducibility
    with (run_dir / "pairs.txt").open("w") as f:
        for pin, pout in pairs:
            f.write(f"IN,{pin}\nOUT,{pout}\n")

    # ---------- Stage 1 ----------
    plays1 = build_all_stage1_plays(pairs, cfg)
    print(f"Stage1 plays total: {len(plays1)}")
    model = NeuralODEModel(cfg).to(cfg.device)

    # Accumulate per-epoch records for a single CSV log
    history_rows = []
    hist1_blocks = []

    if plays1:
        tr1, va1 = train_val_split(plays1, cfg.val_split, cfg.seed)
        T_out_curr = cfg.stage1_start_out

        while True:
            ds_tr = Stage1DynamicDataset(tr1, cfg, T_out_curr)
            ds_va = Stage1DynamicDataset(va1, cfg, T_out_curr)

            # Attach V/A targets (and keep P_tgt from dataset PlayTensors)
            def _collate_stage1(batch):
                plays = [b[0] for b in batch]
                auxs  = [b[1] for b in batch]
                coll = collate_plays(plays)
                B, Nmax, T_out_max, _ = coll['P_tgt'].shape
                V_tgt = torch.zeros(B, Nmax, T_out_max, 2)
                A_tgt = torch.zeros(B, Nmax, T_out_max, 2)
                for b, aux in enumerate(auxs):
                    v = aux['V_tgt']; a = aux['A_tgt']
                    Nt, To = v.shape[:2]
                    V_tgt[b, :Nt, -To:] = v
                    A_tgt[b, :Nt, -To:] = a
                coll['V_tgt'] = V_tgt
                coll['A_tgt'] = A_tgt
                return coll

            loader_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  collate_fn=_collate_stage1)
            loader_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate_stage1)

            print(f"Stage 1 curriculum T_out={T_out_curr}")
            loaders = {"train": loader_tr, "val": loader_va}

            # Use config knobs for warmup vs cap
            epochs = cfg.stage1_epochs if T_out_curr >= cfg.stage1_out_cap else cfg.stage1_warmup_epochs
            best1, hist1 = train(model, loaders, cfg, epochs=epochs, stage="stage1", log_prefix="s1")

            # Log Stage-1 block to history rows (flatten per-epoch)
            for ep_idx, (trv, vv) in enumerate(zip(hist1["train"], hist1["val"]), start=1):
                history_rows.append({
                    "stage": "stage1",
                    "block_T_out": int(T_out_curr),
                    "epoch": int(ep_idx),
                    "train_loss": float(trv),
                    "val_loss": float(vv),
                })
            hist1_blocks.append({"T_out": int(T_out_curr), "hist": hist1})

            if T_out_curr >= cfg.stage1_out_cap:
                break
            T_out_curr = min(cfg.stage1_out_cap, T_out_curr + cfg.stage1_out_step)

        # Save model after Stage 1
        torch.save(model.state_dict(), run_dir / "model_stage1.pt")
        print(f"[run] saved Stage-1 model -> {run_dir / 'model_stage1.pt'}")

    # ---------- Stage 2 ----------
    plays2 = build_all_stage2_plays(pairs, cfg)
    print(f"Stage2 plays total: {len(plays2)}")
    if plays2:
        tr2, va2 = train_val_split(plays2, cfg.val_split, cfg.seed)
        ds_tr2 = PlaysDataset(tr2)
        ds_va2 = PlaysDataset(va2)
        loader_tr2 = DataLoader(ds_tr2, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_plays)
        loader_va2 = DataLoader(ds_va2, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_plays)
        loaders2 = {"train": loader_tr2, "val": loader_va2}
        print("Stage 2 training...")
        best2, hist2 = train(model, loaders2, cfg, epochs=cfg.stage2_epochs, stage="stage2", log_prefix="s2")

        # Log Stage-2 history rows
        for ep_idx, (trv, vv) in enumerate(zip(hist2["train"], hist2["val"]), start=1):
            history_rows.append({
                "stage": "stage2",
                "block_T_out": None,
                "epoch": int(ep_idx),
                "train_loss": float(trv),
                "val_loss": float(vv),
            })

        # Save model after Stage 2
        torch.save(model.state_dict(), run_dir / "model_stage2.pt")
        print(f"[run] saved Stage-2 model -> {run_dir / 'model_stage2.pt'}")

    # ---------- Write CSV log ----------
    if history_rows:
        df_hist = pd.DataFrame(history_rows, columns=["stage","block_T_out","epoch","train_loss","val_loss"])
        df_hist.to_csv(run_dir / "history.csv", index=False)
        print(f"[run] wrote training history -> {run_dir / 'history.csv'}")

if __name__ == "__main__":
    main()
