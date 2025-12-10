# Quick Reference: TD3 vs PPO Commands

## Training Commands

### PPO (RL-Games)
```bash
# Default training
python isaacgymenvs/train.py task=ShadowHand

# With specific config
python isaacgymenvs/train.py \
    task=ShadowHandAsymm \
    train=ShadowHandPPO \
    num_envs=16384
```

### TD3 (SKRL)
```bash
# Default training (edit script to configure)
python isaacgymenvs/td3_train.py
```

---

## Inference/Testing Commands

### PPO (RL-Games)
```bash
# Run inference with checkpoint
python isaacgymenvs/train.py     --config-path=../runs/ShadowHandAsymm_06-16-58-11     --config-name=config     test=True     checkpoint=runs/ShadowHandAsymm_06-16-58-11/nn/ShadowHandAsymm.pth     headless=False     task.env.numEnvs=2     task.task.randomize=True     task.task.randomization_params.actor_params.object.scale.range=[5.0,6.0]
```

### TD3 (SKRL)
```bash
# Option 1: Using improved inference script (RECOMMENDED)
python isaacgymenvs/td3_inference.py \
    --checkpoint runs/ShadowHand_TD3/25-12-06_16-29-35-285982_TD3/checkpoints/best_agent.pt


# TD3: Reduce --num_envs
python isaacgymenvs/td3_inference.py --checkpoint <path> --num_envs 1
```

---

## Graph Visualization
tensorboard --logdir runs/ShadowHand_TD3/25-12-06_16-29-35-285982_TD3/
## Further Documentation

- **SKRL Documentation:** https://skrl.readthedocs.io/
- **RL-Games Documentation:** https://github.com/Denys88/rl_games
