import socket
import json
import re
from collections import deque
from pathlib import Path
import time as pytime

import numpy as np
import requests
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

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

MODEL_PATH = Path("./checkpoints/best_model/best_model.zip")
FALLBACK_MODEL_PATH = Path("./checkpoints/final_model.zip")
ACTION_COOLDOWN = 3.0

DECISION_CFG = {
    "k_interval": 3,
    "cooldown_blocks": 3,
    "cross_ratio_emergency": 0.35,
    "cross_rise_streak": 3,
    "load_emergency": 0.85,
    "load_high_streak": 2,
    "tcl_emergency": 4.0,
    "tcl_worsen_streak": 3,
    "near_full_threshold": 0.95
}


def clip01(x):
    return float(max(0.0, min(1.0, x)))


class UnifiedChainEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.state = np.zeros(12, dtype=np.float32)

    def set_state(self, state_vec):
        self.state = np.array(state_vec, dtype=np.float32)

    def step(self, action):
        reward = self._get_reward(int(action))
        return self.state, reward, False, False, {}

    def _get_reward(self, action):
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
        reward += tps / 120.0
        reward -= cross_ratio * 2.5
        reward -= tcl * 0.8
        reward -= imbalance * 1.2
        reward += inner_tx / 4000.0
        reward -= cross_tx / 5000.0

        if action in [1, 2, 3]:
            reward -= 0.6
        if load > 0.92:
            reward -= 1.5
        elif load < 0.01:
            reward -= 0.1

        if action in [1, 2, 3] and recent_reconfig > 0.5:
            reward -= 0.8
        if action == 4 and cross_ratio > 0.3:
            reward += 0.3
        if action == 5 and cross_ratio > 0.25:
            reward += 0.2
        if action == 3 and imbalance > 0.2:
            reward += 0.4
        if action == 2 and shard_num <= 2:
            reward -= 1.0
        if action == 1 and shard_num >= 8:
            reward -= 1.0

        reward -= broker_ratio * 0.15
        reward -= relay_ratio * 0.12
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
        return np.array([
            clip01(float(state.get("load", 0.0))),
            clip01(float(state.get("tps", 0.0)) / 3000.0),
            clip01(float(state.get("cross_ratio", 0.0))),
            clip01(float(state.get("inner_tx", 0.0)) / 3000.0),
            clip01(float(state.get("cross_tx", 0.0)) / 3000.0),
            clip01(float(state.get("tcl", 0.0)) / 10.0),
            clip01(float(state.get("imbalance", 0.0))),
            clip01(float(state.get("shard_num", 4)) / 8.0),
            0.0,
            clip01(float(state.get("recent_reconfig", 0.0))),
            clip01(float(state.get("broker_ratio", 0.0))),
            clip01(float(state.get("relay_ratio", 0.0))),
        ], dtype=np.float32)


class DecisionGate:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_decision_epoch = {}
        self.cooldown_until = {}
        self.cross_hist = {}
        self.load_hist = {}
        self.tcl_hist = {}

    def _get_hist(self, table, shard_id, maxlen):
        if shard_id not in table:
            table[shard_id] = deque(maxlen=maxlen)
        return table[shard_id]

    def update(self, state):
        shard_id = int(state.get("shard_id", 0))
        self._get_hist(self.cross_hist, shard_id, self.cfg["cross_rise_streak"]).append(float(state.get("cross_ratio", 0.0)))
        self._get_hist(self.load_hist, shard_id, self.cfg["load_high_streak"]).append(float(state.get("load", 0.0)))
        self._get_hist(self.tcl_hist, shard_id, self.cfg["tcl_worsen_streak"]).append(float(state.get("tcl", 0.0)))

    def emergency_triggered(self, state):
        shard_id = int(state.get("shard_id", 0))
        cross_hist = self._get_hist(self.cross_hist, shard_id, self.cfg["cross_rise_streak"])
        load_hist = self._get_hist(self.load_hist, shard_id, self.cfg["load_high_streak"])
        tcl_hist = self._get_hist(self.tcl_hist, shard_id, self.cfg["tcl_worsen_streak"])

        cross_now = float(state.get("cross_ratio", 0.0))
        load_now = float(state.get("load", 0.0))
        tcl_now = float(state.get("tcl", 0.0))

        cross_rising = (
            len(cross_hist) == self.cfg["cross_rise_streak"] and
            all(cross_hist[i] < cross_hist[i + 1] for i in range(len(cross_hist) - 1)) and
            cross_now >= self.cfg["cross_ratio_emergency"]
        )
        load_high = (
            len(load_hist) == self.cfg["load_high_streak"] and
            all(v >= self.cfg["load_emergency"] for v in load_hist)
        )
        tcl_worsening = (
            len(tcl_hist) == self.cfg["tcl_worsen_streak"] and
            all(tcl_hist[i] < tcl_hist[i + 1] for i in range(len(tcl_hist) - 1)) and
            tcl_now >= self.cfg["tcl_emergency"]
        )
        near_full = load_now >= self.cfg["near_full_threshold"]
        return cross_rising or load_high or tcl_worsening or near_full

    def allow_decision(self, state):
        shard_id = int(state.get("shard_id", 0))
        epoch = int(state.get("epoch", 0))

        if epoch < self.cooldown_until.get(shard_id, -1):
            if self.emergency_triggered(state):
                return True, "emergency_in_cooldown"
            return False, "cooldown"

        if self.emergency_triggered(state):
            return True, "emergency"

        last_epoch = self.last_decision_epoch.get(shard_id, None)
        if last_epoch is None:
            return True, "first_decision"

        if epoch - last_epoch >= self.cfg["k_interval"]:
            return True, "interval"

        return False, "wait_interval"

    def on_action_sent(self, state, action):
        shard_id = int(state.get("shard_id", 0))
        epoch = int(state.get("epoch", 0))
        self.last_decision_epoch[shard_id] = epoch
        if action in [1, 2, 3]:
            self.cooldown_until[shard_id] = epoch + self.cfg["cooldown_blocks"]


def extract_json(data):
    match = re.search(rb"\r\n\r\n(.*)", data, re.S)
    return match.group(1) if match else b""


def heuristic_action(state):
    load = float(state.get("load", 0.0))
    tps = float(state.get("tps", 0.0))
    cross_ratio = float(state.get("cross_ratio", 0.0))
    shard_num = int(state.get("shard_num", 4))
    broker_ratio = float(state.get("broker_ratio", 0.0))
    relay_ratio = float(state.get("relay_ratio", 0.0))
    recent_reconfig = float(state.get("recent_reconfig", 0.0))

    if load < 0.01 and tps < 10:
        return 0
    if recent_reconfig > 0.5:
        return 0
    if cross_ratio > 0.45 and broker_ratio < 0.5:
        return 4
    if cross_ratio > 0.30:
        return 3
    if load > 0.85 and shard_num < 8:
        return 1
    if load < 0.20 and cross_ratio < 0.08 and shard_num > 2:
        return 2
    if relay_ratio < 0.5 and cross_ratio > 0.20:
        return 5
    return 0


def safe_action_guard(state, action):
    load = float(state.get("load", 0.0))
    tps = float(state.get("tps", 0.0))
    shard_num = int(state.get("shard_num", 4))
    cross_ratio = float(state.get("cross_ratio", 0.0))
    recent_reconfig = float(state.get("recent_reconfig", 0.0))

    if load < 0.01 and tps < 10:
        return 0
    if recent_reconfig > 0.5 and action in [1, 2, 3]:
        return 0
    if action == 2 and shard_num <= 2:
        return 0
    if action == 1 and shard_num >= 8:
        return 0
    if action in [4, 5] and cross_ratio < 0.10:
        return 0
    return action


def build_action_payload(state, action):
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


def send_action_to_go(payload):
    try:
        requests.post("http://127.0.0.1:9000/rl_action", json=payload, timeout=0.3)
    except Exception:
        pass


def action_text(action):
    return ACTION_TEXT[ACTIONS[action]]


def load_policy(env):
    if MODEL_PATH.exists():
        print(f"加载最优 PPO 模型: {MODEL_PATH}")
        return PPO.load(str(MODEL_PATH), env=env, device="cpu")
    if FALLBACK_MODEL_PATH.exists():
        print(f"加载最终 PPO 模型: {FALLBACK_MODEL_PATH}")
        return PPO.load(str(FALLBACK_MODEL_PATH), env=env, device="cpu")
    print("未找到 best/final 权重，暂时使用启发式策略。")
    return None


if __name__ == "__main__":
    env = UnifiedChainEnv()
    model = load_policy(env)
    feat = FeatureBuilder(window=5)
    gate = DecisionGate(DECISION_CFG)
    last_sent_action = None
    last_sent_ts = 0.0

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 8000))
    server.listen(10)

    print("统一区块链控制服务已启动")
    print("接收状态 -> 块级观测 -> 间隔/紧急触发决策 -> Go执行器")
    print("=" * 80)

    while True:
        conn, addr = server.accept()
        data = conn.recv(65536)
        if not data:
            conn.close()
            continue

        try:
            raw = extract_json(data)
            state = json.loads(raw)

            if bool(state.get("done", False)):
                print("[CTRL] 收到 done=true，本轮仿真结束")
                conn.send(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                conn.close()
                continue

            obs = feat.build(state)
            gate.update(state)
            allow_decision, reason = gate.allow_decision(state)

            if allow_decision:
                if model is None:
                    action = heuristic_action(state)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                action = safe_action_guard(state, int(action))
            else:
                action = 0

            env.set_state(obs)
            reward = env._get_reward(action)
            payload = build_action_payload(state, action)

            now = pytime.time()
            should_send = (
                allow_decision and
                action != 0 and (
                    last_sent_action is None
                    or action != last_sent_action
                    or (now - last_sent_ts) >= ACTION_COOLDOWN
                )
            )

            if should_send:
                send_action_to_go(payload)
                gate.on_action_sent(state, action)
                last_sent_action = action
                last_sent_ts = now

            feat.update(state, action)

            print(
                f"shard={state.get('shard_id')} "
                f"epoch={state.get('epoch', -1)} "
                f"reward={reward:>6} "
                f"load={state.get('load', 0):.2f} "
                f"tps={state.get('tps', 0):>6.1f} "
                f"cross={state.get('cross_ratio', 0):.2f} "
                f"decision={reason} "
                f"action={action_text(action)} "
                f"sent={'Y' if should_send else 'N'}"
            )
        except Exception as e:
            print("错误：", e)

        conn.send(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        conn.close()
