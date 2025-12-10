# BDB 2026 Neural ODE Trajectory Model

This repository contains an experimental trajectory model for the [NFL Big Data Bowl 2026 – Player Trajectory Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview) challenge. The core implementation is in `bdb_unified_pipeline_v2.py`.

## Overview

The model treats player motion as a Neural ODE in phase space \((p, q)\), where:
- \(p\) denotes 2D field position \((x, y)\), and  
- \(q\) denotes 2D velocity \((v_x, v_y)\).

A Velocity–Verlet integrator rolls trajectories forward in time using learned accelerations. Accelerations are predicted from a learned context that combines each player’s individual temporal history with interaction-aware features derived from relative positions.

## Architecture

The pipeline is composed of three main components:

- **TemporalPlayerEncoder**  
  A Transformer encoder operates on each player’s history over time. Inputs include:
  - Dynamic features: position, speed, acceleration, orientation, direction, and velocity,
  - Static/context features: side (offense/defense), coarse role, height and weight.
  
  This produces a per-player context embedding summarizing their trajectory up to the latest observed frame.

- **RPBEncoder (Relative Position Bias Encoder)**  
  Players attend to one another using self-attention with learned relative-position biases. Bias features are based on:
  - Pairwise offsets \((\Delta x, \Delta y)\),
  - Pairwise distance,
  - Same-side indicator (teammate vs opponent).

  This module refines each player’s embedding with interaction-aware context.

- **AccelMLP + Velocity–Verlet Rollout**  
  A small MLP predicts accelerations \((a_x, a_y)\) from:
  - Temporal encoder embeddings,
  - Interaction embeddings from the RPBEncoder,
  - Current velocity,
  - Binned height/weight and side/role embeddings.

  The model then integrates forward in time using a Velocity–Verlet scheme to obtain future positions and velocities.

## Training Pipeline

The script implements a two-stage curriculum:

1. **Stage 1 – Curriculum Pretraining**  
   - Input: variable-length history windows.  
   - Targets: short-horizon future positions, velocities, and accelerations.  
   - Objective: stabilize the Neural ODE dynamics and teach short-term motion.

2. **Stage 2 – Supervised Training on Competition Targets**  
   - Input: full (normalized, left-directed) play histories.  
   - Targets: official challenge outputs (future positions at required frames).  
   - Objective: optimize directly for the evaluation targets while reusing the pretrained dynamics.

Both stages support train/validation splits and log per-epoch RMSE-based metrics to a CSV in a timestamped `runs/run_*/` directory, together with the configuration snapshot and model checkpoints.

## Notes

- Data loading assumes preprocessed CSVs following the competition format, with separate “input” and “output” files per training split.
- All plays are mirrored so that play direction is standardized to “left” before feature extraction.
- This is research code and is not affiliated with or endorsed by the NFL or Kaggle.
