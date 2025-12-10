
import isaacgym
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env 
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.memories.torch.random import RandomMemory
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.noises.torch import GaussianNoise
from skrl.trainers.torch import SequentialTrainer
import torch
import torch.nn as nn

# Define the Actor (Policy) Model
class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # ShadowHand "full_state" has 211 observations
        # ShadowHand has 20 actions
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, self.num_actions),
            nn.Tanh()  # TD3 actions are usually in [-1, 1]
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# Define the Critic Model (Twin Critics will use instances of this)
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
        # Critic takes state and action as input
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

if __name__ == '__main__':
    # 1. Load the Isaac Gym environment
    # task_name="ShadowHand" ensures it loads the config from isaacgymenvs/cfg/task/ShadowHand.yaml
    # num_envs is reduced to 256 (vs 16384 in PPO) for off-policy efficiency
    env = load_isaacgym_env_preview4(
        task_name="ShadowHand",
        num_envs=256,
        headless=True,  # Set to False to see the simulation window
        # sim_device="cuda:0",
        # rl_device="cuda:0",
        # physics_engine="physx",
    )

    # 2. Wrap env so spaces/actions are gymnasium-compatible  ✅
    env = wrap_env(env)

    # 2. Instantiate models
    models = {}
    models["policy"] = Actor(env.observation_space, env.action_space, env.device)
    models["target_policy"] = Actor(env.observation_space, env.action_space, env.device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, env.device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, env.device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, env.device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, env.device)

    # 3. Configure Memory (Replay Buffer)
    # memory_size is number of steps per environment.
    # 10,000 steps * 256 envs = 2,560,000 total transitions
    memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=env.device, replacement=False)

    # 4. Configure TD3 settings
    cfg = TD3_DEFAULT_CONFIG.copy()
    
    # Exploration Noise (Gaussian) - Critical fix: must be an object, not a float
    cfg["exploration"]["noise"] = GaussianNoise(mean=0, std=0.2, device=env.device)
    
    cfg["batch_size"] = 4096               # Larger batch size for stability
    cfg["random_timesteps"] = 1000         # Random steps before learning
    cfg["learning_starts"] = 1000
    cfg["gradient_steps"] = 1              # Updates per step
    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["directory"] = "runs/ShadowHand_TD3"
    cfg["experiment"]["checkpoint_interval"] = 100000  # Save checkpoint every 1 million steps

    agent = TD3(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)

    # 5. Train
    # Total timesteps = 5M
    trainer = SequentialTrainer(cfg={"timesteps": 1000000}, env=env, agents=agent)
    trainer.train()
