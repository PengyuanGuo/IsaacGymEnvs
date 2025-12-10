"""
TD3 Inference Script with Command-Line Arguments
Run trained TD3 agents on ShadowHand environment

Usage:
    python isaacgymenvs/td3_inference.py --checkpoint runs/ShadowHand_TD3/*/checkpoints/best_agent.pt
    python isaacgymenvs/td3_inference.py --checkpoint runs/ShadowHand_TD3/*/checkpoints/agent_500000.pt --num_envs 4 --headless
"""

import argparse
import isaacgym
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env 
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.models.torch import Model, DeterministicMixin
import torch
import torch.nn as nn
import os
import sys

# Define the Actor (Policy) Model
class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # ShadowHand "full_state" has 211 observations, 20 actions
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, self.num_actions),
            nn.Tanh()
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# Define Critic (needed for agent initialization, won't be used in inference)
class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

def parse_args():
    parser = argparse.ArgumentParser(description="TD3 Inference for ShadowHand")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., runs/ShadowHand_TD3/*/checkpoints/best_agent.pt)"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no visualization)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of steps to run (default: infinite)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ShadowHand",
        help="Task name (default: ShadowHand)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    checkpoint_path = args.checkpoint
    num_envs = args.num_envs
    headless = args.headless
    max_steps = args.max_steps
    task = args.task
    print("=" * 80)
    print("TD3 Inference Configuration")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Task: {task}")
    print(f"Number of Environments: {num_envs}")
    print(f"Headless: {headless}")
    print(f"Max Steps: {max_steps if max_steps else 'Infinite'}")
    print("=" * 80)
    # Clear sys.argv to prevent Hydra from seeing our arguments
    sys.argv = [sys.argv[0]]  # Keep only the script name
    
    # 1. Load Environment
    print("\nLoading environment...")
    env = load_isaacgym_env_preview4(
        task_name=task,
        num_envs=num_envs,
        headless=headless
    )
    env = wrap_env(env)
    
    print(f"✓ Environment loaded successfully")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # 2. Create models
    print("\nInitializing models...")
    models = {}
    models["policy"] = Actor(env.observation_space, env.action_space, env.device)
    models["target_policy"] = Actor(env.observation_space, env.action_space, env.device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, env.device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, env.device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, env.device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, env.device)
    
    # 3. Create agent
    cfg = TD3_DEFAULT_CONFIG.copy()
    cfg["experiment"]["write_interval"] = 0  # Disable logging during inference
    
    agent = TD3(
        models=models,
        memory=None,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device
    )
    
    # 4. Load checkpoint
    print(f"\nLoading checkpoint...")
    agent.load(checkpoint_path)
    print(f"✓ Checkpoint loaded successfully")
    
    # 5. Set evaluation mode
    agent.set_mode("eval")
    
    # 6. Run inference
    print("\n" + "=" * 80)
    print("Starting inference...")
    print("Press Ctrl+C to stop")
    print("=" * 80 + "\n")
    
    states, infos = env.reset()
    step_count = 0
    episode_rewards = torch.zeros(num_envs, device=env.device)
    episode_lengths = torch.zeros(num_envs, device=env.device)
    completed_episodes = 0
    
    with torch.no_grad():
        try:
            while True:
                # Get action from agent
                actions = agent.act(states, timestep=0, timesteps=0)[0]
                
                # Step environment
                states, rewards, terminated, truncated, infos = env.step(actions)
                
                # Track statistics (flatten rewards to ensure correct shape)
                episode_rewards += rewards.flatten()
                episode_lengths += 1
                step_count += 1
                
                # Handle episode completion
                dones = (terminated | truncated).flatten()
                if dones.any():
                    for i in range(num_envs):
                        if dones[i]:
                            completed_episodes += 1
                            print(f"Episode {completed_episodes} completed:")
                            print(f"  Environment {i}: Reward = {episode_rewards[i]:.2f}, Length = {int(episode_lengths[i])}")
                            episode_rewards[i] = 0
                            episode_lengths[i] = 0
                
                # Check max steps
                if max_steps and step_count >= max_steps:
                    print(f"\nReached maximum steps ({max_steps}). Stopping...")
                    break
                
                # Print progress every 1000 steps
                if step_count % 1000 == 0:
                    print(f"Steps: {step_count}, Episodes completed: {completed_episodes}")
                    
        except KeyboardInterrupt:
            print("\n\nInference stopped by user")
    
    print("\n" + "=" * 80)
    print("Inference Summary")
    print("=" * 80)
    print(f"Total steps: {step_count}")
    print(f"Episodes completed: {completed_episodes}")
    print("=" * 80)

if __name__ == '__main__':
    main()
