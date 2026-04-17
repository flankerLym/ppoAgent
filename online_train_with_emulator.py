import argparse
import json
import queue
import re
import socket
import threading
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


DEFAULT_CONFIG = {
    "seed": 42,
    "device": "cpu",
    "server": {
        "host": "127.0.0.1",
        "port": 8000,
        "recv_bytes": 65536,
        "reuse_addr": True,
        "go_action_url": "http://127.0.0.1:9000/rl_action"
    },
    "training": {
        "total_timesteps": 100000,
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "gamma": 0.99,
        "verbose": 1,
        "checkpoint_freq": 5000,
        "checkpoints_dir": "./checkpoints"
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
        "relay_usage_penalty": 0.12
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


def extract_json(data: bytes) -> bytes:
    match = re.search(rb"\r\n\r\n(.*)", data, re.S)
    return match.group(1) if match else b""


class StateReceiver(threading.Thread):
    def __init__(self, host: str, port: int, recv_bytes: int, reuse_addr: bool = True):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.recv_bytes = recv_bytes
        self.reuse_addr = reuse_addr
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop_flag = threading.Event()
        self.server = socket.socket()
        if self.reuse_addr:
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(10)

    def run(self):
        while not self._stop_flag.is_set():
            try:
                conn, _addr = self.server.accept()
                data = conn.recv(self.recv_bytes)
                if data:
                    raw = extract_json(data)
                    if raw:
                        try:
                            state = json.loads(raw)
                            self.queue.put(state)
                        except Exception:
                            pass
                try:
                    conn.send(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                except Exception:
                    pass
                conn.close()
            except OSError:
                break

    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def get(self, timeout=None) -> Dict[str, Any]:
        if timeout is None:
            return self.queue.get()
        return self.queue.get(timeout=timeout)

    def stop(self):
        self._stop_flag.set()
        try:
            self.server.close()
        except Exception:
            pass


class ManualEmulatorOnlineEnv(gym.Env):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.server_cfg = cfg["server"]
        self.reward_cfg = cfg["reward"]
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.receiver = StateReceiver(
            self.server_cfg["host"],
            int(self.server_cfg["port"]),
            int(self.server_cfg["recv_bytes"]),
            bool(self.server_cfg["reuse_addr"]),
        )
        self.receiver.start()
        self.current_state = np.zeros(12, dtype=np.float32)

    def _state_to_obs(self, state: Dict[str, Any]) -> np.ndarray:
        return np.array([
            float(max(0.0, min(1.0, state.get("load", 0.0)))),
            float(max(0.0, min(1.0, state.get("tps", 0.0) / 3000.0))),
            float(max(0.0, min(1.0, state.get("cross_ratio", 0.0)))),
            float(max(0.0, min(1.0, state.get("inner_tx", 0.0) / 3000.0))),
            float(max(0.0, min(1.0, state.get("cross_tx", 0.0) / 3000.0))),
            float(max(0.0, min(1.0, state.get("tcl", 0.0) / 10.0))),
            float(max(0.0, min(1.0, state.get("imbalance", 0.0)))),
            float(max(0.0, min(1.0, state.get("shard_num", 4) / 8.0))),
            0.0,
            float(max(0.0, min(1.0, state.get("recent_reconfig", 0.0)))),
            float(max(0.0, min(1.0, state.get("broker_ratio", 0.0)))),
            float(max(0.0, min(1.0, state.get("relay_ratio", 0.0)))),
        ], dtype=np.float32)

    def _reward(self, obs: np.ndarray, action: int) -> float:
        c = self.reward_cfg
        load = float(obs[0])
        tps = float(obs[1]) * 3000.0
        cross_ratio = float(obs[2])
        inner_tx = float(obs[3]) * 3000.0
        cross_tx = float(obs[4]) * 3000.0
        tcl = float(obs[5]) * 10.0
        imbalance = float(obs[6])
        shard_num = float(obs[7]) * 8.0
        recent_reconfig = float(obs[9])
        broker_ratio = float(obs[10])
        relay_ratio = float(obs[11])

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

    def _send_action(self, action: int):
        actions = {
            0: "noop",
            1: "split_shard",
            2: "merge_shard",
            3: "trigger_clpa",
            4: "enable_broker",
            5: "enable_relay",
            6: "enter_cooldown",
        }
        payload = {
            "action_id": int(action),
            "action_name": actions[action],
            "shard_id": 0,
            "params": {
                "delta": 1 if action == 1 else (-1 if action == 2 else 0),
                "cooldown_rounds": 3 if action == 6 else 0,
            },
        }
        try:
            requests.post(self.server_cfg["go_action_url"], json=payload, timeout=0.5)
        except Exception:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.receiver.clear()
        print("[TRAIN] 等待你手动启动 emulator，并发送第一条 state ...")
        first_state = self.receiver.get(timeout=None)
        print("[TRAIN] 已收到第一条 state")
        self.current_state = self._state_to_obs(first_state)
        return self.current_state, {}

    def step(self, action):
        self._send_action(int(action))
        reward = self._reward(self.current_state, int(action))

        print("[TRAIN] 等待下一条 state ...")
        next_state = self.receiver.get(timeout=None)
        print("[TRAIN] 已收到下一条 state")
        self.current_state = self._state_to_obs(next_state)

        terminated = False
        truncated = False
        return self.current_state, reward, terminated, truncated, {}

    def close(self):
        self.receiver.stop()


def load_config(path: str) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG
    p = Path(path)
    if p.exists():
        user_cfg = json.loads(p.read_text(encoding="utf-8"))
        cfg = deep_update(DEFAULT_CONFIG, user_cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="emulator_train_config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_dir = Path(cfg["training"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = ManualEmulatorOnlineEnv(cfg)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(cfg["training"]["learning_rate"]),
        n_steps=int(cfg["training"]["n_steps"]),
        batch_size=int(cfg["training"]["batch_size"]),
        gamma=float(cfg["training"]["gamma"]),
        verbose=int(cfg["training"]["verbose"]),
        device=str(cfg["device"]),
        seed=int(cfg["seed"]),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(cfg["training"]["checkpoint_freq"]),
        save_path=str(ckpt_dir),
        name_prefix="ppo_controller"
    )

    try:
        model.learn(
            total_timesteps=int(cfg["training"]["total_timesteps"]),
            callback=checkpoint_callback,
        )
        model.save(str(ckpt_dir / "final_model.zip"))
        print(f"最终模型已保存到: {ckpt_dir / 'final_model.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
