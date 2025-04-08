from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3 import DQN
import os

# 1. Register your custom environment
register(
    id="my-highway-v0",
    entry_point="my_envs.custom_highway:MyHighwayEnv",
)

# 2. Create the environment
env = gym.make("my-highway-v0")

# 3. Create a directory for TensorBoard logs (optional but recommended)
log_dir = "custom_highway_dqn/"
os.makedirs(log_dir, exist_ok=True)

# 4. Instantiate the DQN model with TensorBoard logging enabled
model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=15000,
    learning_starts=200,
    batch_size=32,
    gamma=0.8,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=50,
    verbose=1,
    tensorboard_log=log_dir  # This is where logs will be stored
)

# 5. Train the model and specify a tb_log_name
#    This name will create a subfolder in "custom_highway_dqn/"
model.learn(
    total_timesteps=int(10e4),
    tb_log_name="DQN_highway_run5"  # You can change this name for different runs
)

# 6. Save the model
model.save(os.path.join(log_dir, "model"))

# 7. Close the environment
env.close()
