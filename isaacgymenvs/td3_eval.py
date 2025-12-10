import isaacgym
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env 
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.models.torch import Model, DeterministicMixin
from skrl.trainers.torch import SequentialTrainer
import torch
import torch.nn as nn
import sys

# Define the Actor (Policy) Model
class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # self.num_observations is automatically set by Model.__init__ from observation_space
        # It should be 211 for ShadowHand "full_state"
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

if __name__ == '__main__':
    # Checkpoint path
    checkpoint_path = "runs/ShadowHand_TD3/25-12-06_16-29-35-285982_TD3/checkpoints/best_agent.pt"

    # 1. Load Environment (Headless = False for visualization)
    env = load_isaacgym_env_preview4(
        task_name="ShadowHand",
        num_envs=1, 
        headless=False 
    )
    env = wrap_env(env)
    
    print("Environment Observation Space:", env.observation_space)
    print("Environment Action Space:", env.action_space)

    # 2. Instantiate Model
    models = {}
    models["policy"] = Actor(env.observation_space, env.action_space, env.device)
    
    # Placeholders
    models["target_policy"] = models["policy"] 
    models["critic_1"] = models["policy"]
    models["critic_2"] = models["policy"]
    models["target_critic_1"] = models["policy"]
    models["target_critic_2"] = models["policy"]

    # 3. Configure Agent
    cfg = TD3_DEFAULT_CONFIG.copy()
    cfg["experiment"]["write_interval"] = 0
    
    agent = TD3(models=models,
                memory=None,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)

    # 4. Load Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)

    # 5. Evaluation Loop
    agent.set_mode("eval")
    
    print("Starting visualization...")
    states, infos = env.reset()
    
    with torch.no_grad():
        while True:
            # Agent action
            actions = agent.act(states, timestep=0, timesteps=0)[0]
            
            # Step environment
            states, rewards, terminated, truncated, infos = env.step(actions)