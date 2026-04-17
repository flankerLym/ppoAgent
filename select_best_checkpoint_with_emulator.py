import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


DEFAULT_CONFIG = {
    "server": {
        "host": "127.0.0.1",
        "port": 8000,
        "go_action_url": "http://127.0.0.1:9000/rl_action"
    },
    "emulator": {
        "workdir": "E:/keyan/code/block-emulator",
        "command": ["go", "run", "main.go", "-g", "-S", "4", "-N", "3"]
    },
    "checkpoints_dir": "./checkpoints",
    "eval_runs_per_ckpt": 2,
    "service_script": "./serve_best_model_with_emulator.py",
    "service_config": "./emulator_eval_config.json"
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG
    p = Path(path)
    if p.exists():
        user_cfg = json.loads(p.read_text(encoding="utf-8"))
        cfg = deep_update(DEFAULT_CONFIG, user_cfg)
    return cfg


def list_checkpoints(ckpt_dir: Path) -> List[Path]:
    files = sorted(ckpt_dir.glob("ppo_controller_*.zip"))
    if (ckpt_dir / "final_model.zip").exists():
        files.append(ckpt_dir / "final_model.zip")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="emulator_eval_config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_dir = Path(cfg["checkpoints_dir"])
    best_dir = ckpt_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list_checkpoints(ckpt_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    # 这里先给出“可运行的筛选框架”
    # 当前用一个简单代理分数：优先更晚 checkpoint。
    # 你后续可以把这里替换成“每个 ckpt 启服务 + 启 emulator + 读实验日志分数”的真实评估。
    scored: List[Tuple[float, Path]] = []
    for i, ckpt in enumerate(checkpoints):
        score = float(i + 1)
        scored.append((score, ckpt))
        print(f"checkpoint={ckpt.name} score={score:.3f}")

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = scored[0]
    target = best_dir / "best_model.zip"
    shutil.copy2(best_path, target)

    summary = {
        "best_checkpoint": str(best_path),
        "best_score": best_score,
        "copied_to": str(target),
    }
    (best_dir / "best_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
