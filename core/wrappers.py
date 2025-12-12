"""
Gymnasium-compatible environment wrappers and make_env helper.

Features:
- MaxAndSkip wrapper (frame skipping + max-pool)
- Grayscale + Resize (for pixel envs)
- FrameStack (stack last k frames along channel dim)
- Reward clipping
- Optional normalization (simple running-mean/var)
- make_env(env_id, seed, frame_stack=4, width=84, height=84, clip_reward=True, skip=4)
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TransformObservation
import cv2
import collections
import math


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame (frames are max-pooled over last two frames)."""
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        assert skip >= 1
        self._skip = skip
        self._obs_buffer = None

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == 0:
                self._obs_buffer = obs
            else:
                # For pixel envs we want max over last two frames to reduce flicker
                try:
                    self._obs_buffer = np.maximum(self._obs_buffer, obs)
                except Exception:
                    self._obs_buffer = obs
            total_reward += reward
            if done:
                break
        return self._obs_buffer, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer = obs
        return obs, info


class ResizeGrayScaleObs(gym.ObservationWrapper):
    """Convert RGB to grayscale and resize to (height, width). Returns uint8 image."""
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        # observation space becomes (H, W) grayscale
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
        )

    def observation(self, obs):
        # obs could be dict with 'image' key or raw image
        if isinstance(obs, dict):
            # try to find image in dict
            img = obs.get("rgb", None) or obs.get("image", None) or obs.get("pixels", None)
            if img is None:
                raise ValueError("Received dict observation but no image found under keys.")
        else:
            img = obs

        # if image has channel last
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 2:
            # already grayscale
            img = img
        else:
            # fallback: convert to uint8 and reshape if needed
            img = np.asarray(img)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        resized = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)


class FrameStack(gym.Wrapper):
    """Stack k last frames along channel dimension (returns H x W x k for pixel envs)"""
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        if isinstance(env.observation_space, spaces.Box):
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 2:
                h, w = obs_shape
                self.observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(h, w, k),
                    dtype=env.observation_space.dtype,
                )
            elif len(obs_shape) == 3:
                h, w, c = obs_shape
                self.observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(h, w, c * k),
                    dtype=env.observation_space.dtype,
                )
            else:
                raise ValueError("Unsupported observation shape for FrameStack.")
        else:
            # For non-box observation spaces, we won't modify space (useful for vector envs)
            self.observation_space = env.observation_space

        self.frames = collections.deque(maxlen=k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        # stack along last axis
        arrs = list(self.frames)
        try:
            stacked = np.concatenate([np.expand_dims(a, -1) if a.ndim == 2 else a for a in arrs], axis=-1)
        except Exception:
            # fallback to np.stack
            stacked = np.stack(arrs, axis=-1)
        return stacked


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to -1, 0, +1 (common in Atari experiments)."""
    def __init__(self, env: gym.Env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def reward(self, reward):
        return float(np.clip(reward, self.low, self.high))


class NormalizeObservation(gym.ObservationWrapper):
    """Simple running mean/std normalization for float observations (non-image)."""
    def __init__(self, env: gym.Env, eps: float = 1e-8):
        super().__init__(env)
        shape = env.observation_space.shape
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = eps

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        self.count += 1.0
        alpha = 1.0 / self.count
        self.mean = (1.0 - alpha) * self.mean + alpha * obs
        self.var = (1.0 - alpha) * self.var + alpha * (obs - self.mean) ** 2
        std = np.sqrt(self.var) + 1e-8
        return (obs - self.mean) / std


def is_pixel_observation(env: gym.Env) -> bool:
    """Heuristic: treat Box with 3 channels or 2D HxW as pixel env."""
    obs_space = env.observation_space
    if not isinstance(obs_space, spaces.Box):
        return False
    shape = obs_space.shape
    if len(shape) == 3 and shape[2] in (1, 3):
        return True
    if len(shape) == 2:
        return True
    return False


def make_env(
    env_id: str,
    seed: Optional[int] = None,
    frame_stack: int = 4,
    width: int = 84,
    height: int = 84,
    clip_reward: bool = True,
    skip: int = 4,
    normalize: bool = False,
) -> gym.Env:
    """
    Create an environment with common wrappers for RL experiments.
    - For pixel environments: MaxAndSkip -> ResizeGrayScaleObs -> FrameStack
    - For low-dim envs (LunarLander): optionally normalize observations
    """
    env = gym.make(env_id, render_mode=None)  # remove render for training
    if seed is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        except TypeError:
            pass

    # apply frame skipping for speed (useful for Atari)
    if skip and skip > 1:
        env = MaxAndSkipEnv(env, skip=skip)

    if is_pixel_observation(env):
        # convert to grayscale + resize
        env = ResizeGrayScaleObs(env, width, height)
        # stack frames
        if frame_stack and frame_stack > 1:
            env = FrameStack(env, frame_stack)
        if clip_reward:
            env = ClipRewardEnv(env, -1.0, 1.0)
    else:
        # low-dim observation spaces (like LunarLander) - no image transforms
        if normalize:
            env = NormalizeObservation(env)
        if clip_reward:
            # for control tasks sometimes clipping is optional; default: no clipping for LunarLander
            pass

    return env