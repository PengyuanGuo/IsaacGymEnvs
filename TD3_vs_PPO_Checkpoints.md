# TD3 vs PPO Checkpoint Differences

## Overview
The checkpoint formats differ significantly between TD3 (trained with `td3_train.py`) and PPO (trained with `train.py`) due to the fundamental differences in the algorithms and frameworks used.

## Key Differences

### 1. **Framework Differences**
- **TD3**: Uses [SKRL](https://skrl.readthedocs.io/) framework
- **PPO**: Uses [RL-Games](https://github.com/Denys88/rl_games) framework

### 2. **Checkpoint Structure**

#### TD3 Checkpoint (`.pt` files)
Located in: `runs/ShadowHand_TD3/25-12-06_16-29-35-285982_TD3/checkpoints/`

**Contains:**
- `policy`: Actor network state_dict (the main policy for inference)
- `target_policy`: Target actor network state_dict
- `critic_1`: First critic network state_dict
- `critic_2`: Second critic network state_dict
- `target_critic_1`: Target first critic state_dict
- `target_critic_2`: Target second critic state_dict
- `policy_optimizer`: Policy optimizer state
- `critic_optimizer`: Critic optimizer state

**Key characteristics:**
- Multiple networks stored separately (actor-critic architecture)
- Includes target networks (for stable training)
- Direct state_dict format for each network
- Saved via SKRL's checkpoint mechanism

#### PPO Checkpoint (`.pth` files)
Located in: `runs/ShadowHandAsymm_06-16-58-11/nn/`

**Contains:**
- `model`: Combined actor-critic network state_dict
- `central_val_stats`: Statistics for centralized value function
- `assymetric_vf_nets`: Asymmetric value function networks
- `optimizer`: Optimizer state
- `epoch`: Training epoch number
- `frame`: Total frames/timesteps
- `last_mean_rewards`: Recent reward statistics
- `env_state`: Environment state information

**Key characteristics:**
- Single combined model for actor-critic
- Includes training metadata (epoch, frame count)
- Asymmetric actor-critic support
- Saved via RL-Games checkpoint mechanism

### 3. **Network Architecture Differences**

#### TD3 Architecture (from `td3_train.py`)
```python
Actor (Policy):
- Input: 211 observations (ShadowHand full_state)
- Hidden: [400, 300] with ReLU
- Output: 20 actions with Tanh
- Separate deterministic policy network

Critic:
- Input: 211 observations + 20 actions
- Hidden: [400, 300] with ReLU
- Output: 1 Q-value
- Twin critics for double Q-learning
```

#### PPO Architecture (from RL-Games config)
```python
Combined Actor-Critic:
- Input: Observation space (varies by config)
- Hidden: Configurable MLP units
- Output: Action distribution (stochastic) + Value estimate
- Single network with shared features
```

## How to Run TD3 Inference

### Method 1: Using the Existing `td3_eval.py` Script

The repository already has a TD3 evaluation script at `isaacgymenvs/td3_eval.py`:

```bash
# Activate your conda environment
conda activate ece_595

# Run the evaluation script (modify the checkpoint path inside if needed)
cd /home/pengyuan/2025_Summer/ece_595/IsaacGymEnvs
python isaacgymenvs/td3_eval.py
```

The script:
1. Loads the ShadowHand environment with visualization (`headless=False`)
2. Creates the Actor network matching the training architecture
3. Loads the checkpoint using `agent.load(checkpoint_path)`
4. Runs inference loop displaying the trained policy

**Key points in `td3_eval.py`:**
- Line 33: Checkpoint path (modify to use different checkpoints)
- Line 39: `headless=False` enables visualization
- Line 38: `num_envs=1` for single environment evaluation
- Line 73: `agent.set_mode("eval")` sets evaluation mode
- Line 81: `agent.act(states, ...)` performs inference

### Method 2: Manual Checkpoint Loading

```python
import torch
import isaacgym
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env 
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from td3_train import Actor  # Import your Actor class

# Load checkpoint
checkpoint_path = "runs/ShadowHand_TD3/25-12-06_16-29-35-285982_TD3/checkpoints/best_agent.pt"
checkpoint = torch.load(checkpoint_path)

# Create environment
env = load_isaacgym_env_preview4(
    task_name="ShadowHand",
    num_envs=1,
    headless=False
)
env = wrap_env(env)

# Create and load actor
actor = Actor(env.observation_space, env.action_space, env.device)
actor.load_state_dict(checkpoint["policy"])
actor.eval()

# Run inference
states, _ = env.reset()
with torch.no_grad():
    while True:
        actions = actor({"states": states})[0]
        states, rewards, terminated, truncated, infos = env.step(actions)
```

### Method 3: Using Different Checkpoints

The TD3 training saves checkpoints at regular intervals:
- `best_agent.pt`: Best performing checkpoint (recommended)
- `agent_100000.pt`, `agent_200000.pt`, ...: Checkpoints at 100k step intervals
- `agent_1000000.pt`: Final checkpoint

To use a different checkpoint, simply change the path:
```python
checkpoint_path = "runs/ShadowHand_TD3/25-12-06_16-29-35-285982_TD3/checkpoints/agent_500000.pt"
```

## Comparing PPO and TD3 Inference

### PPO Inference (via `train.py`)
```bash
python isaacgymenvs/train.py \
    task=ShadowHandAsymm \
    test=True \
    checkpoint=runs/ShadowHandAsymm_06-16-58-11/nn/ShadowHandAsymm.pth
```

### TD3 Inference (via `td3_eval.py`)
```bash
python isaacgymenvs/td3_eval.py
```

## Why Are They Different?

1. **Algorithm Design**:
   - TD3 is an **off-policy** algorithm requiring separate policy and critic networks
   - PPO is an **on-policy** algorithm with a combined actor-critic network

2. **Framework Choice**:
   - SKRL (TD3) focuses on modular agent design with explicit model separation
   - RL-Games (PPO) uses integrated training with combined models

3. **Training Requirements**:
   - TD3 needs target networks for stability (hence 6 networks total)
   - PPO only needs current networks plus optional asymmetric critic

4. **Checkpoint Philosophy**:
   - SKRL saves complete agent state (all networks + optimizers)
   - RL-Games saves complete training state (model + training metadata)

## Quick Reference

| Aspect | TD3 | PPO |
|--------|-----|-----|
| Framework | SKRL | RL-Games |
| File Extension | `.pt` | `.pth` |
| Location | `checkpoints/` | `nn/` |
| Networks | Separate (6 total) | Combined (1-2 total) |
| Inference Script | `td3_eval.py` | `train.py test=True` |
| Loading Method | `agent.load()` | Via RL-Games runner |
| Key for Policy | `"policy"` | `"model"` |

## Troubleshooting

### Issue: "Architecture mismatch when loading checkpoint"
- **Solution**: Ensure the Actor class definition matches the one used during training
- The network architecture (layer sizes) must be identical

### Issue: "Missing keys in state_dict"
- **Solution**: You're trying to load a TD3 checkpoint with PPO code or vice versa
- Use the correct evaluation script for each algorithm

### Issue: "CUDA out of memory during inference"
- **Solution**: Reduce `num_envs` to 1 in the evaluation script
- Or use CPU: modify device settings in the script

## Additional Notes

- TD3 checkpoints are generally smaller than PPO checkpoints
- TD3 uses deterministic actions during inference (no exploration noise)
- PPO can use either deterministic or stochastic actions depending on config
- For best results, use `best_agent.pt` which is saved when the agent achieves the highest reward



