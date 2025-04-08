import os
import shutil
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DQN

# Register your custom environment
register(
    id="my-highway-v0",
    entry_point="my_envs.custom_highway:MyHighwayEnv",
)


def main():
    # Create the environment
    env = gym.make("my-highway-v0", render_mode="rgb_array", config={
        "collision_termination": False,
        "offroad_terminal": False,
        "duration": 1000,
    })

    # Load the trained DQN model
    model = DQN.load("custom_highway_dqn/model.zip")

    # Reset the environment
    obs, info = env.reset()
    done = False
    step_count = 0

    # Run for a maximum of 50 steps (or until done/truncated)
    while not done and step_count < 50:
        step_count += 1

        # Predict action using the trained RL model
        action, _ = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)

        # Print states for all vehicles (assuming they are stored in env.unwrapped.road.vehicles)
        print(f"Step {step_count}:")
        for i, veh in enumerate(env.unwrapped.road.vehicles):
            # veh.position is assumed to be a numpy array [x, y]
            x, y = veh.position
            v = veh.speed
            # Mark the ego vehicle (if it's the controlled vehicle)
            tag = " (ego)" if veh is env.unwrapped.vehicle else ""
            print(f"  Vehicle {i}{tag}: x = {x:.2f}, y = {y:.2f}, v = {v:.2f}")

        if truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
