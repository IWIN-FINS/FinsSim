### 速度控制

```bash
export DISPLAY=:0
python -m scripts.train --config=ppo_control_for_velocity --exp-name="ppo_control_for_local_velocity" --port-offset=2000 --overwrite
```

```bash
python -m scripts.eval --config=ppo_control_for_velocity --exp-name="ppo_control_for_local_velocity"
```

```bash
export DISPLAY=:0
python -m scripts.train --config=ppo_control_for_velocity_no_rotation --exp-name="ppo_control_for_local_velocity_no_rotation" --env-base-port=20000 --overwrite --num-envs=32
```

```bash
python -m scripts.eval --config=ppo_control_for_velocity_no_rotation --exp-name="ppo_control_for_local_velocity_no_rotation"
```

debug only

```bash
export DISPLAY=:0
python -m scripts.train --config=ppo_control_for_velocity_no_rotation --exp-name="ppo_control_for_local_velocity_no_rotation" --env-base-port=2000 --overwrite --num_envs=2 --debug
```
