# Atari DQN Agents (TF-Agents): Seaquest & Space Invaders

Deep Q-Network (DQN) agents trained with **TF-Agents** to play **Seaquest** and **Space Invaders**.  
The repo provides a shared, reproducible pipeline: preprocessing, CNN Q-network, ε-greedy policy with replay, training/resume scripts, evaluation curves, and gameplay videos.

---

## Introduction

Each Atari game is modeled as an MDP with:
- **State:** stack of 4 grayscale frames
- **Action:** discrete game actions
- **Reward:** clipped to **[-1, 1]** for stability

We use an ε-greedy policy, experience replay, a target network, and RMSProp optimization.  
(Design and settings summarized from the project report and training scripts.

---

## Objectives
- Implement a **shared DQN architecture** for both games.
- Train for a **computationally feasible** number of steps and evaluate progress.
- **Resume** partially trained runs and **export policies**.
- Produce **training curves** and **gameplay videos** for qualitative assessment.

---

## Approach

### Environment & Preprocessing
- Atari envs via `suite_atari.load(...)` with standard wrappers and **frame stacking**.
- **Reward clipping** wrapper and **ActionRepeat(times=4)**.
- Normalize pixels in-network: `obs -> float32 / 255.0`.

### Q-Network (shared)
- Conv layers: `(32, 8×8, stride 4) → (64, 4×4, stride 2) → (64, 3×3, stride 1)`
- Fully connected: `512` units
- Loss: **Huber**
- Optimizer: **RMSProp** (centered)

### Training Strategy
- Replay buffer: `100k`, batch `128`
- Target update every **2000** steps
- ε-greedy with **polynomial decay** `1.0 → 0.01`
- Warm-up: **20k** random steps, then train (e.g., **50k** steps per run; script supports resuming to reach 100k+)

### Evaluation
- Average return & episode length tracked during training
- Curve plotting + **policy export per checkpoint**
- Offline evaluation & **video recording** at selected checkpoints (poor / mid / best)

---

## How to Run

> Tip: Use a Python 3.10+ environment with GPU for faster training/inference.

### 1) Install dependencies
    pip install "tensorflow==2.11.*" tf-agents==0.14.0 gym==0.23.0 \
                imageio imageio-ffmpeg matplotlib numpy opencv-python \
                autorom==0.4.2 autorom-accept-rom-license==0.6.1

### 2) Install Atari ROMs (one-time)
    AutoROM --accept-license

### 3) Train a game
- **Seaquest**
    
        python buildAndTrainAgent_Seaquest.py
- **Space Invaders**
    
        python buildAndTrainAgent_SpaceInvaders.py

(Each script sets up envs, warm-up collection, trains DQN, saves checkpoints and policies.)

### 4) Resume training (optional)
Resumes **+50k** steps for any game with saved checkpoints.

    python resumeTraining.py


### 5) Create training curves & gameplay videos
Evaluates saved policies at multiple checkpoints, saves **Avg Return vs Steps** plots, and records **poor / intermediate / best** videos per game in `videos/`.

    python createTrainingCurvesAndVideos.py


### 6) Deploy a single game player (latest checkpoint)
Records one full-episode video for the latest saved policy of each game.

    python deployGamePlayer.py


---

## Results (Summary)

- With ~**100k** steps per game (50k train + 50k resume), both agents **learn** non-trivial policies:
  - **Seaquest:** steadily improving average return and longer episodes.
  - **Space Invaders:** slower improvement, but stable learning trajectory.
- More steps (1M+) would further improve returns; this project emphasizes a **compact, reproducible** baseline.

---

## Files in this Repository

- `buildAndTrainAgent_Seaquest.py` — Train DQN for Seaquest (env, Q-net, replay, training loop, checkpoints, policy export).
- `buildAndTrainAgent_SpaceInvaders.py` — Train DQN for Space Invaders (same pipeline as Seaquest).
- `resumeTraining.py` — Resume training from saved checkpoints (+50k steps).
- `createTrainingCurvesAndVideos.py` — Evaluate multiple checkpoints, plot **Avg Return vs Steps**, and save **poor/mid/best** gameplay videos.
- `deployGamePlayer.py` — Load the **latest** saved policy per game and record one gameplay video.
- `assignment4_runfile.ipynb` — Notebook containing runnable code blocks and experiments.
- `Atari_DQN_TFAgents.pdf` — Report with architecture, training strategy, results, and analysis.

---

## Tools & Versions

- **TensorFlow 2.11**, **TF-Agents 0.14**, **Gym 0.23**
- **AutoROM** (with license accept), **ImageIO (+ ffmpeg)**, **Matplotlib**, **OpenCV**, **NumPy**

---

## Author

**Aakash Maskara**  
*M.S. Robotics & Autonomy, Drexel University*  
Reinforcement Learning | Deep Learning

[LinkedIn](https://linkedin.com/in/aakashmaskara) • [GitHub](https://github.com/AakashMaskara)
