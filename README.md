# Challenge 4 — GAIL on ALE/MsPacman-v5
**Group 7 | Machine Learning — Universidad Distrital Francisco José de Caldas**

Three-algorithm comparison: DQN (Challenge 1) → PPO (Challenge 3) → GAIL (Challenge 4).

---

## Overview

This challenge implements **Generative Adversarial Imitation Learning (GAIL)**
(Ho & Ermon, 2016) and compares it against the best DQN from Challenge 1 and
PPO from Challenge 3 on `ALE/MsPacman-v5`.

**Group 7 hypothesis:** Ms. Pac-Man has relatively dense rewards.
GAIL may converge more slowly than PPO/DQN because the adversarial reward is
a proxy rather than the true game signal. **BC alone** — which imitates good
movement patterns from demonstrations — may be competitive with full GAIL or
even PPO on this denser-reward game.

---

## Project structure

```
challenge4/group7/
├── utils.py          # make_env, AtariActorCritic, compute_gae
├── discriminator.py  # GAILDiscriminator CNN
├── collect_demos.py  # Collect (s, a) pairs from the DQN checkpoint
├── train_bc.py       # Behavioral Cloning baseline
├── train_gail.py     # GAIL training loop (PPO as inner RL)
├── evaluate.py       # Three-way comparison evaluation
├── pyproject.toml    # Dependencies (mirrors Challenge 1)
├── README.md
└── CHECKLIST.md
```

---

## Setup

```bash
# Reuse the Challenge 1 virtual environment (all dependencies are identical)
cd ../Challenge_1
poetry install

# Or create a new one here
poetry install
```

---

## Reproduction: step-by-step

### Step 1 — Collect demonstrations

```bash
# 50k steps from exp_02_lr_high (best DQN, mean reward ≈ 1747)
python collect_demos.py \
    --model-path ../Challenge_1/models/mspacman_dqn \
    --n-steps 50000 \
    --output demos_50k.npz

# 5k and 20k for the demonstration-size ablation
python collect_demos.py --n-steps 5000  --output demos_5k.npz
python collect_demos.py --n-steps 20000 --output demos_20k.npz
```

### Step 2 — Train the BC baseline

```bash
python train_bc.py --demos-path demos_50k.npz --n-epochs 30
```

### Step 3 — Train GAIL

```bash
# Full run (Group 7 starter configuration)
python train_gail.py --demos-path demos_50k.npz --total-steps 5000000

# Reduced budget (if compute is limited — document the constraint)
python train_gail.py --demos-path demos_50k.npz --total-steps 2000000

# Ablation: small demonstration set
python train_gail.py --demos-path demos_5k.npz  --save-path models/gail_5k.pt
```

### Step 4 — Evaluate and compare

```bash
# Individual evaluation (10 deterministic episodes)
python evaluate.py --mode bc   --model-path models/bc_policy.pt
python evaluate.py --mode gail --model-path models/gail_policy.pt
python evaluate.py --mode dqn  --model-path ../Challenge_1/models/mspacman_dqn

# Side-by-side three-way table
python evaluate.py --compare \
    --bc-path   models/bc_policy.pt \
    --gail-path models/gail_policy.pt \
    --dqn-path  ../Challenge_1/models/mspacman_dqn \
    --n-episodes 10
```

### Step 5 — TensorBoard

```bash
# All runs together
tensorboard --logdir logs --port 6006
```

---

## Hyperparameter summary

| Component | Parameter | Value | Rationale |
|---|---|---|---|
| **GAIL** | total_steps | 5 000 000 | Budget parity with PPO |
| | horizon | 1 024 | PPO rollout length |
| | lr_policy | 2.5e-4 | Standard PPO Atari lr |
| | lr_disc | 3e-4 | Slightly faster disc convergence |
| | disc_updates | 3 | Per rollout; avoids over-training disc |
| | ent_coef (λ) | 0.01 | Entropy bonus for residual exploration |
| | gamma | 0.99 | Standard Atari discount |
| **BC** | n_epochs | 30 | Thorough supervised training |
| | lr | 1e-4 | Standard Adam lr |
| **Demos** | source | exp_02_lr_high | Best DQN (mean ≈ 1747) |
| | n_steps | 50 000 | Primary dataset |
| | n_steps (ablation) | 5 000 / 20 000 | Quantity sensitivity study |

---

## Demonstration ablation

Two dataset sizes are tested per Challenge 4 requirements:

| Dataset | Steps | Expected effect |
|---|---|---|
| `demos_5k.npz` | 5 000 | Small; less diverse; may cause discriminator collapse |
| `demos_50k.npz` | 50 000 | Large; diverse maze states; primary training set |

---

## Analysis questions (to be answered in the paper)

1. Does GAIL outperform pure RL (DQN, PPO) on MsPacman's dense rewards?
2. How sensitive is GAIL to demonstration quality and quantity?
3. Does the discriminator remain informative, or does it collapse to D ≈ 0.5?
4. In what regime is BC alone competitive with full GAIL or PPO?

---

## References

- Ho & Ermon (2016). *Generative Adversarial Imitation Learning*. NeurIPS.
- Mnih et al. (2015). *Human-level control through deep RL*. Nature.
- Schulman et al. (2017). *Proximal Policy Optimization*. arXiv.
- Fu et al. (2018). *Learning Robust Rewards with Adversarial IRL*. ICLR.
