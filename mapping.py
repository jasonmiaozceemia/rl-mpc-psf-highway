import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DQN

register(
    id="my-highway-v0",
    entry_point="my_envs.custom_highway:MyHighwayEnv",
)

def interpret_rl_action(rl_action_idx):
    """
    Interpret the RL action index according to HighwayEnv's default discrete meta actions.
    By default, the mapping is:
        0 -> LANE_LEFT
        1 -> IDLE
        2 -> LANE_RIGHT
        3 -> FASTER
        4 -> SLOWER
    """
    try:
        idx = int(rl_action_idx)
    except Exception as e:
        print("Error converting RL action to int:", e)
        idx = rl_action_idx
    mapping = {
        0: "LANE_LEFT",
        1: "IDLE",
        2: "LANE_RIGHT",
        3: "FASTER",
        4: "SLOWER"
    }
    return mapping.get(idx, "UNKNOWN")

def main():

    env = gym.make("my-highway-v0")

    model = DQN.load("custom_highway_dqn/model.zip")

    obs, info = env.reset()
    done = False
    step_count = 0

    while not done and step_count < 50:
        step_count += 1
        rl_action, _ = model.predict(obs, deterministic=True)
        action_meaning = interpret_rl_action(rl_action)
        print(f"Step {step_count}: RL action index = {rl_action} -> {action_meaning}")
        obs, reward, done, truncated, info = env.step(rl_action)

    env.close()

if __name__ == "__main__":
    main()
