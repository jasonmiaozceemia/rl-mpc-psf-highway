from gymnasium.envs.registration import register
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
import os
from matplotlib import pyplot as plt

# Register the same environment again for this separate script
register(
    id="my-highway-v0",
    entry_point="my_envs.custom_highway:MyHighwayEnv",
)

# Create the environment for video recording
env = gym.make("my-highway-v0", render_mode="rgb_array")

# Set up a folder to save the videos
video_folder = "videos"
os.makedirs(name=video_folder, exist_ok=True)

video_env = RecordVideo(
    env=env,
    video_folder=video_folder,
    name_prefix="videos",
    disable_logger=True
)

try:
    video_env.unwrapped.set_record_video_wrapper(video_env)
except AttributeError:
    pass

# Load the trained model
model = DQN.load("custom_highway_dqn/model.zip")
#model = DQN.load("custom_highway_dqn/model")
obs, info = video_env.reset()
plt.imshow(env.render())
plt.show()

# Run the model in a loop and generate the video
for episode in range(1):  # Record one episode
    #obs, info = video_env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = video_env.step(action)

plt.imshow(env.render())
plt.show()
video_env.close()
env.close()
print("Video recorded in:", video_folder)