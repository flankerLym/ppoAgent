import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG = {
    "workdir_emulator": "E:/keyan/code/block-emulator",
    "workdir_agent": "E:/keyan/code/ppoAgent",
    "service_script": "serve_best_model_with_emulator.py",
    "service_config": "emulator_eval_config.json",
    "runs": [
        {
            "name": "baseline_static",
            "agent_mode": "none",
            "emulator_command": ["go", "run", "main.go", "-g", "-S", "4", "-N", "3"]
        },
        {
            "name": "method_best_model",
            "agent_mode": "best_model",
            "emulator_command": ["go", "run", "main.go", "-g", "-S", "4", "-N", "3"]
        },
        {
            "name": "ablation_no_best_fallback_heuristic",
            "agent_mode": "heuristic",
            "emulator_command": ["go", "run", "main.go", "-g", "-S", "4", "-N", "3"]
        }
    ]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiment_suite.example.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent_dir = Path(cfg["workdir_agent"])
    emu_dir = Path(cfg["workdir_emulator"])

    print("实验矩阵：")
    print(json.dumps(cfg["runs"], indent=2, ensure_ascii=False))
    print("\n注意：本 runner 负责统一拉起 agent / emulator 进程。")
    print("正式指标建议仍以 block-emulator 现有 result/log 输出为准。")

    for run in cfg["runs"]:
        print("\n" + "=" * 80)
        print(f"开始实验: {run['name']}")
        print("=" * 80)

        agent_proc = None
        if run["agent_mode"] in ["best_model", "heuristic"]:
            agent_cmd = ["python", cfg["service_script"], "--config", cfg["service_config"]]
            agent_proc = subprocess.Popen(agent_cmd, cwd=str(agent_dir))
            time.sleep(3)

        emu_proc = subprocess.Popen(run["emulator_command"], cwd=str(emu_dir))
        emu_proc.wait()

        if agent_proc is not None and agent_proc.poll() is None:
            agent_proc.kill()
            try:
                agent_proc.wait(timeout=5)
            except Exception:
                pass

        print(f"完成实验: {run['name']}")


if __name__ == "__main__":
    main()
