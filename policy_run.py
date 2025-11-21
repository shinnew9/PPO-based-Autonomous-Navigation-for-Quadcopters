# Revised some code here since the original code is based on unstable assumption that the image file is always ready
import os
import gym
import yaml
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper



# 0) Reset
class ResetCompat(gym.Wrapper):
    def reset(self, *args, **kwargs):
        # AirSimDroneEnv는 인자 없는 reset만 지원하니까 싹 무시
        obs = self.env.reset()   # Our AirSim env only returns obs
        return obs, {}           # Match (obs, info) of what SB3 expects


class StepCompat(gym.Wrapper):
    def step(self, action):
        out = self.env.step(action)
        # If it's the old version, the length would be 4
        if len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
            return obs, reward, terminated, truncated, info
        # If it's new formula, it would be new
        return out

# 1) changes the channel automatically (N, H, W, C) -> (N, C, H, W) 하고, 공간도 바꿔주는 Wrapper
class ChannelLastToFirst(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        # Original Space: (50, 50, 3)
        old_space = venv.observation_space
        assert isinstance(old_space, spaces.Box)
        h, w, c = old_space.shape
        # SB3 model이 기대하는 형태로 공간을 바꿔준다: (C, H, W)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(c, h, w),
            dtype=old_space.dtype,
        )

    def reset(self):
        obs = self.venv.reset()   # (n_env, H, W, C)
        return self._transpose(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._transpose(obs), rewards, dones, infos

    def _transpose(self, obs):
        # (n_env, H, W, C) -> (n_env, C, H, W)
        return np.transpose(obs, (0, 3, 1, 2))


# 2) Get train environment configs
with open('scripts/config.yml', 'r') as f:
    cfg = yaml.safe_load(f)
train_cfg = cfg.get("TrainEnv", {})


# 3) Function that makes env (Format that DummyVecEnv is expecting)
def make_env():
    # Copy all the necessary id from the originial repo
    env = gym.make(
        "scripts:test-env-v0", 
        ip_address="127.0.0.1",  # Make sure TrainEnv.exe is always turned on when runnign this py file
        image_shape=(50, 50, 3),   # Match with config.yml
        env_config=train_cfg
    )
    env = ResetCompat(env)
    env = StepCompat(env)
    env = Monitor(env)
    return env


env = DummyVecEnv([make_env])
# Model expects (3, 50, 50), transpose to appropriate size
env = ChannelLastToFirst(env)

print("👀 final observation_space:", env.observation_space)


# 4) Model Load
model_path = os.path.join("saved_policy", "ppo_navigation_policy")
# 이 레포는 보통 ppo_navigation_policy.zip 으로 저장돼 있을거라서 둘 다 체크
if os.path.exists(model_path + ".zip"):
    model_path = model_path + ".zip"
elif not os.path.exists(model_path):
    raise FileNotFoundError(f"Cannot find the model file : {model_path}(.zip)")


# 5) Covering SB3 model
custom_objects = {
    "lr_schedule": lambda _: 3e-4,  # set this as constant   
    "clip_range": 0.2,              # Also set Cliping as a constant 
}
model = PPO.load(model_path, env=None, custom_objects=custom_objects)


# Run the trained policy
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done.any():
        obs = env.reset()
