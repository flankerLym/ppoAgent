import argparse
import json
import re
import socket
import time as pytime
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from stable_baselines3 import PPO


DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 8000,
        "recv_bytes": 4096,
        "reuse_addr": True,
        "go_action_url": "http://127.0.0.1:9000/rl_action",
        "allow_auto_fallback_port": False
    },
    "policy": {
        "device": "cpu",
        "deterministic": True,
        "load_order": [
            "./checkpoints/best_model/best_model.zip",
            "./checkpoints/final_model.zip",
            "./ppo_chain_controller.zip"
        ]
    },
    "features": {
        "window": 5
    },
    "actions": {
        "cooldown_seconds": 3.0
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
    },
    "heuristic": {
        "idle_load": 0.01,
        "idle_tps": 10.0,
        "recent_reconfig_block": 0.5,
        "enable_broker_cross_ratio": 0.45,
        "trigger_clpa_cross_ratio": 0.30,
        "split_load": 0.85,
        "merge_load": 0.20,
        "merge_cross_ratio": 0.08,
        "enable_relay_cross_ratio": 0.20,
        "broker_ratio_limit": 0.5,
        "relay_ratio_limit": 0.5,
        "min_shards": 2,
        "max_shards": 8
    }
}

ACTIONS = {
    0: "noop",
    1: "split_shard",
    2: "merge_shard",
    3: "trigger_clpa",
    4: "enable_broker",
    5: "enable_relay",
    6: "enter_cooldown",
}

ACTION_TEXT = {
    "noop": "不操作",
    "split_shard": "拆分分片",
    "merge_shard": "合并分片",
    "trigger_clpa": "触发CLPA重分区",
    "enable_broker": "启用Broker模式",
    "enable_relay": "启用Relay模式",
    "enter_cooldown": "进入冷却期",
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


class UnifiedChainEnv(gym.Env):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.reward_cfg = cfg["reward"]
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.state = np.zeros(12, dtype=np.float32)

    def set_state(self, state_vec):
        self.state = np.array(state_vec, dtype=np.float32)

    def step(self, action):
        reward = self._get_reward(int(action))
        return self.state, reward, False, False, {}

    def _get_reward(self, action: int):
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
        return round(float(reward), 3)

    def reset(self, seed=None, options=None):
        self.state = np.zeros(12, dtype=np.float32)
        return self.state, {}


class FeatureBuilder:
    def __init__(self, window=5):
        self.window = window
        self.reconfig_hist = deque(maxlen=window)

    def update(self, state, action=0):
        self.reconfig_hist.append(1.0 if action in [1, 2, 3] else 0.0)

    def build(self, state):
        return np.array(
            [
                clip01(float(state.get("load", 0.0))),
                clip01(float(state.get("tps", 0.0)) / 3000.0),
                clip01(float(state.get("cross_ratio", 0.0))),
                clip01(float(state.get("inner_tx", 0.0)) / 3000.0),
                clip01(float(state.get("cross_tx", 0.0)) / 3000.0),
                clip01(float(state.get("tcl", 0.0)) / 10.0),
                clip01(float(state.get("imbalance", 0.0))),
                clip01(float(state.get("shard_num", 4)) / 8.0),
                0.0,
                clip01(float(sum(self.reconfig_hist) > 0)),
                clip01(float(state.get("broker_ratio", 0.0))),
                clip01(float(state.get("relay_ratio", 0.0))),
            ],
            dtype=np.float32,
        )


def extract_json(data: bytes) -> bytes:
    match = re.search(rb"\r\n\r\n(.*)", data, re.S)
    return match.group(1) if match else b""


def heuristic_action(state: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    h = cfg["heuristic"]
    load = float(state.get("load", 0.0))
    tps = float(state.get("tps", 0.0))
    cross_ratio = float(state.get("cross_ratio", 0.0))
    shard_num = int(state.get("shard_num", 4))
    broker_ratio = float(state.get("broker_ratio", 0.0))
    relay_ratio = float(state.get("relay_ratio", 0.0))
    recent_reconfig = float(state.get("recent_reconfig", 0.0))

    if load < float(h["idle_load"]) and tps < float(h["idle_tps"]):
        return 0
    if recent_reconfig > float(h["recent_reconfig_block"]):
        return 0
    if cross_ratio > float(h["enable_broker_cross_ratio"]) and broker_ratio < float(h["broker_ratio_limit"]):
        return 4
    if cross_ratio > float(h["trigger_clpa_cross_ratio"]):
        return 3
    if load > float(h["split_load"]) and shard_num < int(h["max_shards"]):
        return 1
    if load < float(h["merge_load"]) and cross_ratio < float(h["merge_cross_ratio"]) and shard_num > int(h["min_shards"]):
        return 2
    if relay_ratio < float(h["relay_ratio_limit"]) and cross_ratio > float(h["enable_relay_cross_ratio"]):
        return 5
    return 0


def safe_action_guard(state: Dict[str, Any], action: int, cfg: Dict[str, Any]) -> int:
    h = cfg["heuristic"]
    load = float(state.get("load", 0.0))
    tps = float(state.get("tps", 0.0))
    shard_num = int(state.get("shard_num", 4))
    cross_ratio = float(state.get("cross_ratio", 0.0))
    recent_reconfig = float(state.get("recent_reconfig", 0.0))

    if load < float(h["idle_load"]) and tps < float(h["idle_tps"]):
        return 0
    if recent_reconfig > float(h["recent_reconfig_block"]) and action in [1, 2, 3]:
        return 0
    if action == 2 and shard_num <= int(h["min_shards"]):
        return 0
    if action == 1 and shard_num >= int(h["max_shards"]):
        return 0
    if action in [4, 5] and cross_ratio < 0.10:
        return 0
    return action


def build_action_payload(state: Dict[str, Any], action: int):
    name = ACTIONS[action]
    return {
        "action_id": int(action),
        "action_name": name,
        "shard_id": int(state.get("shard_id", 0)),
        "params": {
            "delta": 1 if name == "split_shard" else (-1 if name == "merge_shard" else 0),
            "cooldown_rounds": 3 if name == "enter_cooldown" else 0,
        },
    }


def send_action_to_go(payload: Dict[str, Any], cfg: Dict[str, Any]):
    try:
        requests.post(cfg["server"]["go_action_url"], json=payload, timeout=0.3)
    except Exception:
        pass


def action_text(action: int) -> str:
    return ACTION_TEXT[ACTIONS[action]]


def load_config(path: str) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG
    if path:
        p = Path(path)
        if p.exists():
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            cfg = deep_update(DEFAULT_CONFIG, user_cfg)
    return cfg


def load_best_or_fallback(env: gym.Env, cfg: Dict[str, Any]) -> Optional[PPO]:
    load_order = cfg["policy"]["load_order"]
    for path in load_order:
        p = Path(path)
        if p.exists():
            print(f"加载模型: {p}")
            return PPO.load(str(p), env=env, device=cfg["policy"]["device"])
    print("未找到 best/final/model 权重，暂时使用启发式策略。")
    return None


def bind_server(server: socket.socket, cfg: Dict[str, Any]) -> Tuple[str, int]:
    host = str(cfg["server"]["host"])
    port = int(cfg["server"]["port"])
    allow_fallback = bool(cfg["server"]["allow_auto_fallback_port"])
    try:
        server.bind((host, port))
        return host, port
    except PermissionError:
        if not allow_fallback:
            raise
        alt_port = port + 1
        server.bind((host, alt_port))
        print(f"端口 {port} 绑定失败，已自动切换到 {alt_port}")
        return host, alt_port


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="service_controller_config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = UnifiedChainEnv(cfg)
    model = load_best_or_fallback(env, cfg)
    feat = FeatureBuilder(window=int(cfg["features"]["window"]))
    last_sent_action = None
    last_sent_ts = 0.0

    server = socket.socket()
    if bool(cfg["server"]["reuse_addr"]):
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    host, port = bind_server(server, cfg)
    server.listen(10)

    print("统一区块链控制服务已启动")
    print(f"接收状态 -> AI决策 -> Go执行器 监听: {host}:{port}")
    print("=" * 80)

    while True:
        conn, _addr = server.accept()
        data = conn.recv(int(cfg["server"]["recv_bytes"]))
        if not data:
            conn.close()
            continue

        try:
            raw = extract_json(data)
            state = json.loads(raw)

            obs = feat.build(state)

            if model is None:
                action = heuristic_action(state, cfg)
            else:
                action, _ = model.predict(obs, deterministic=bool(cfg["policy"]["deterministic"]))
                action = int(action)

            action = safe_action_guard(state, int(action), cfg)

            env.set_state(obs)
            reward = env._get_reward(action)
            payload = build_action_payload(state, action)

            now = pytime.time()
            cooldown = float(cfg["actions"]["cooldown_seconds"])
            should_send = (
                action != 0 and (
                    last_sent_action is None
                    or action != last_sent_action
                    or (now - last_sent_ts) >= cooldown
                )
            )

            if should_send:
                send_action_to_go(payload, cfg)
                last_sent_action = action
                last_sent_ts = now

            feat.update(state, action)

            print(
                f"shard={state.get('shard_id')} "
                f"reward={reward:>6} "
                f"load={state.get('load', 0):.2f} "
                f"tps={state.get('tps', 0):>6.1f} "
                f"cross={state.get('cross_ratio', 0):.2f} "
                f"action={action_text(action)} "
                f"sent={'Y' if should_send else 'N'}"
            )
        except Exception as e:
            print("错误：", e)

        conn.send(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        conn.close()


if __name__ == "__main__":
    main()
