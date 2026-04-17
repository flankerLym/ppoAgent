# PPOAgent + block-emulator 联调方案

这套文件用于：
1. 让 `block-emulator` 作为在线环境，训练 PPO。
2. 定期把权重保存到 `checkpoints/`。
3. 从所有 checkpoint 中筛选 best model。
4. 用 best model 启动控制服务，回接仿真平台跑实验。
5. 跑基线 / 方法对比 / 消融实验。

## 关键前提

你的 `block-emulator` 必须已经接入以下两点：
- `supervisor.go` 能把 RL state 推到 Python 监听端口（默认 `127.0.0.1:8000`）
- `StartRLActionServer()` 能接收 `POST /rl_action`

也就是你当前 Go 端至少要具备：
- state -> Python
- action <- Python

如果这两个没打通，训练脚本无法工作。

## 文件说明

- `online_train_with_emulator.py`
  在线训练 PPO，仿真器作为环境，定期存 checkpoint
- `select_best_checkpoint_with_emulator.py`
  遍历 checkpoint，跑仿真评估，选 best_model.zip
- `serve_best_model_with_emulator.py`
  用 best model 启动控制服务，供仿真平台做正式实验
- `run_experiment_suite.py`
  方法对比与消融实验 runner
- `emulator_train_config.json`
  训练配置
- `emulator_eval_config.json`
  评估 / 服务配置
- `experiment_suite.example.json`
  实验矩阵示例

## 推荐流程

### 1. 在线训练
```bash
python online_train_with_emulator.py --config emulator_train_config.json
```

### 2. 从 checkpoints 里选 best
```bash
python select_best_checkpoint_with_emulator.py --config emulator_eval_config.json
```

### 3. 用 best model 启动控制服务
```bash
python serve_best_model_with_emulator.py --config emulator_eval_config.json
```

### 4. 跑方法对比 / 消融
```bash
python run_experiment_suite.py --config experiment_suite.example.json
```

## 重要提醒

- 这套方案会在训练 / 评估时 **启动 emulator 子进程**
- 训练速度会明显慢于 synthetic debug env
- 建议你先在小规模配置上确认链路通，再开长训练
- Windows 下如果 8000/9000 端口占用，请先释放
