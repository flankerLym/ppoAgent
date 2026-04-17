import argparse
import json
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "cpu",
    "checkpoints_dir": "./checkpoints",
    "tensorboard_log": "./tensorboard_logs",
    "resume_from": "",
    "save_freq": 5000,
    "eval_freq": 5000,
    "total_timesteps": 100000,
    "policy": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "verbose": 1,
    },
    "reward": {
        "tps_scale": 120.0,
        "cross_penalty": 2.5,
        "tcl_penalty": 0.8,
        "imbalance_penalty": 1.2,
        "inner_bonus_div": 4000.0,
        "cross_cost_div": 5000.0,
        "reconfig_penalty": 0.6,
        "recent_reconfig_penalty": 0.8,
        "overload_penalty": 1.5,
        "underload_penalty": 0.1,
        "broker_bonus": 0.3,
        "relay_bonus": 0.2,
        "clpa_bonus": 0.4,
        "small_shard_merge_penalty": 1.0,
        "large_shard_split_penalty": 1.0,
        "broker_usage_penalty": 0.15,
        "relay_usage_penalty": 0.12,
    },
    "state_sampler": {
        "load": [0.0, 1.0],
        "tps": [0.0, 3000.0],
        "cross_ratio": [0.0, 1.0],
        "inner_tx": [0.0, 3000.0],
        "cross_tx": [0.0, 3000.0],
        "tcl": [0.0, 10.0],
        "imbalance": [0.0, 1.0],
        "shard_num": [2, 8],
        "recent_reconfig": [0.0, 1.0],
        "broker_ratio": [0.0, 1.0],
        "relay_ratio": [0.0, 1.0]
    }
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


class UnifiedChainEnv(gym.Env):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.reward_cfg = cfg["reward"]
        self.sampler = cfg["state_sampler"]
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.state = np.zeros(12, dtype=np.float32)

    def sample_state(self) -> np.ndarray:
        s = self.sampler
        load = np.random.uniform(*s["load"])
        tps = np.random.uniform(*s["tps"]) / 3000.0
        cross_ratio = np.random.uniform(*s["cross_ratio"])
        inner_tx = np.random.uniform(*s["inner_tx"]) / 3000.0
        cross_tx = np.random.uniform(*s["cross_tx"]) / 3000.0
        tcl = np.random.uniform(*s["tcl"]) / 10.0
        imbalance = np.random.uniform(*s["imbalance"])
        shard_num = np.random.randint(int(s["shard_num"][0]), int(s["shard_num"][1]) + 1) / 8.0
        recent_reconfig = np.random.uniform(*s["recent_reconfig"])
        broker_ratio = np.random.uniform(*s["broker_ratio"])
        relay_ratio = np.random.uniform(*s["relay_ratio"])
        return np.array(
            [load, tps, cross_ratio, inner_tx, cross_tx, tcl, imbalance, shard_num, 0.0, recent_reconfig, broker_ratio, relay_ratio],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.sample_state()
        return self.state, {}

    def step(self, action):
        reward = self._get_reward(int(action))
        self.state = self.sample_state()
        return self.state, reward, False, False, {}

    def _get_reward(self, action: int) -> float:
        c = self.reward_cfg
        load = float(self.state[0])
        tps = float(self.state[1]) * 3000.0
        cross_ratio = float(self.state[2])
        inner_tx = float(self.state[3]) * 3000.0
        cross_tx = float(self.state[4]) * 3000.0
        tcl = float(self.state[5]) * 10.0
        imbalance = float(self.state[6])
        shard_num = float(self.state[7]) * 8.0
        recent_reconfig = float(self.state[9])
        broker_ratio = float(self.state[10])
        relay_ratio = float(self.state[11])

        reward = 0.0
        reward += tps / float(c["tps_scale"])
        reward -= cross_ratio * float(c["cross_penalty"])
        reward -= tcl * float(c["tcl_penalty"])
        reward -= imbalance * float(c["imbalance_penalty"])
        reward += inner_tx / float(c["inner_bonus_div"])
        reward -= cross_tx / float(c["cross_cost_div"])

        if action in [1, 2, 3]:
            reward -= float(c["reconfig_penalty"])
        if load > 0.92:
            reward -= float(c["overload_penalty"])
        elif load < 0.01:
            reward -= float(c["underload_penalty"])

        if action in [1, 2, 3] and recent_reconfig > 0.5:
            reward -= float(c["recent_reconfig_penalty"])
        if action == 4 and cross_ratio > 0.3:
            reward += float(c["broker_bonus"])
        if action == 5 and cross_ratio > 0.25:
            reward += float(c["relay_bonus"])
        if action == 3 and imbalance > 0.2:
            reward += float(c["clpa_bonus"])
        if action == 2 and shard_num <= 2:
            reward -= float(c["small_shard_merge_penalty"])
        if action == 1 and shard_num >= 8:
            reward -= float(c["large_shard_split_penalty"])

        reward -= broker_ratio * float(c["broker_usage_penalty"])
        reward -= relay_ratio * float(c["relay_usage_penalty"])
        return float(reward)


def load_config(path: str) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG
    if path:
        p = Path(path)
        if p.exists():
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            cfg = deep_update(DEFAULT_CONFIG, user_cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_controller_config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_dir = Path(cfg["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "best_model").mkdir(parents=True, exist_ok=True)

    env = UnifiedChainEnv(cfg)
    eval_env = UnifiedChainEnv(cfg)

    p = cfg["policy"]
    resume_from = str(cfg.get("resume_from", "")).strip()

    if resume_from and Path(resume_from).exists():
        print(f"继续训练已有权重: {resume_from}")
        model = PPO.load(resume_from, env=env, device=cfg["device"])
    else:
        print("开始训练（注意：这是 synthetic debug env，用于先把 checkpoint/best-model 机制跑通）")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=float(p["learning_rate"]),
            n_steps=int(p["n_steps"]),
            batch_size=int(p["batch_size"]),
            gamma=float(p["gamma"]),
            verbose=int(p["verbose"]),
            device=str(cfg["device"]),
            seed=int(cfg["seed"]),
            tensorboard_log=str(cfg["tensorboard_log"]),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(cfg["save_freq"]),
        save_path=str(ckpt_dir),
        name_prefix="ppo_controller"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir / "best_model"),
        log_path=str(ckpt_dir / "best_model"),
        eval_freq=int(cfg["eval_freq"]),
        deterministic=True,
        render=False
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    model.learn(
        total_timesteps=int(cfg["total_timesteps"]),
        reset_num_timesteps=not bool(resume_from and Path(resume_from).exists()),
        callback=callbacks,
    )

    final_path = ckpt_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"最终模型已保存到: {final_path}")
    print(f"最优模型目录: {ckpt_dir / 'best_model'}")
    print("通常实验加载路径应优先用 checkpoints/best_model/best_model.zip")


if __name__ == "__main__":
    main()
