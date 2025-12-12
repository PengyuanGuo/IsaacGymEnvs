# Shadow Hand Block 尺寸随机化训练指南

## 概述

本指南说明如何在使用 PPO 算法训练 Shadow Hand 抓取 Block 任务时启用 Block 尺寸的随机化(Domain Randomization)。

## 配置说明

### 1. 启用域随机化

在配置文件 `isaacgymenvs/cfg/task/ShadowHand.yaml` 中,已经包含了完整的域随机化配置,包括 Block 尺寸随机化。

**关键配置位置**: 第 63-74 行

```yaml
task:
  randomize: False  # <-- 将此项改为 True 以启用域随机化
  randomization_params:
    # ... 其他配置 ...
    actor_params:
      # ... hand 配置 ...
      object:
        scale:
          range: [0.95, 1.05]  # Block 尺寸缩放范围 (95%-105%)
          operation: "scaling"
          distribution: "uniform"
          setup_only: True
```

### 2. 启用方法

有两种方式启用 Block 尺寸随机化:

#### 方法 1: 修改 YAML 配置文件(推荐)

编辑 `isaacgymenvs/cfg/task/ShadowHand.yaml`:

```yaml
task:
  randomize: True  # 改为 True
```

然后正常启动训练:

```bash
python train.py task=ShadowHand
```

#### 方法 2: 使用命令行参数(不修改文件)

直接在命令行启动时覆盖配置:

```bash
python train.py task=ShadowHand task.randomize=True
```

### 3. 自定义 Block 尺寸范围

如果想调整 Block 的尺寸范围,可以使用命令行参数:

```bash
# 示例:尺寸范围从 90% 到 110%
python train.py task=ShadowHand \
    task.randomize=True \
    task.randomization_params.actor_params.object.scale.range=[0.9,1.1]
```

```bash
# 示例:尺寸范围从 80% 到 120% (更大的变化)
python train.py task=ShadowHand \
    task.randomize=True \
    task.randomization_params.actor_params.object.scale.range=[0.8,1.2]
```

### 4. 完整训练命令示例

```bash
# 基础训练 + Block 尺寸随机化
python train.py task=ShadowHand task.randomize=True

# 指定环境数量
python train.py task=ShadowHand task.randomize=True num_envs=8192

# 多 GPU 训练
python train.py task=ShadowHand task.randomize=True multi_gpu=True

# 自定义尺寸范围 + 更多设置
python train.py task=ShadowHand \
    task.randomize=True \
    task.randomization_params.actor_params.object.scale.range=[0.85,1.15] \
    num_envs=16384 \
    headless=True
```

## 域随机化参数详解

### Block (Object) 相关随机化

配置文件中 `object` 部分的随机化参数:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `scale.range` | 尺寸缩放范围 | `[0.95, 1.05]` (±5%) |
| `scale.operation` | 操作类型 | `"scaling"` (缩放) |
| `scale.distribution` | 分布类型 | `"uniform"` (均匀分布) |
| `scale.setup_only` | 仅在初始化时随机化 | `True` |
| `rigid_body_properties.mass.range` | 质量范围 | `[0.5, 1.5]` (50%-150%) |
| `rigid_shape_properties.friction.range` | 摩擦力范围 | `[0.7, 1.3]` (70%-130%) |

### 重要说明

- **`setup_only: True`**: 每个环境的 Block 尺寸在初始化时随机确定,训练过程中保持不变
- **`distribution: "uniform"`**: 尺寸在指定范围内均匀分布
- **`operation: "scaling"`**: 对原始尺寸进行缩放操作

### 其他可用的随机化

启用 `task.randomize=True` 后,以下参数也会同时随机化:

1. **Shadow Hand 机械臂**:
   - 腱(tendon)的阻尼和刚度
   - 关节(DOF)的阻尼和刚度
   - 刚体质量
   - 摩擦系数

2. **物理环境**:
   - 重力 (±0.4 m/s²)

3. **观测和动作**:
   - 观测噪声
   - 动作噪声

## 验证随机化是否生效

### 方法 1: 查看训练日志

启动训练后,如果看到类似以下信息说明随机化已启用:

```
Applying randomization to: object
Randomizing scale with range [0.95, 1.05]
```

### 方法 2: 启用可视化

```bash
# 不使用 headless 模式,可以直观看到不同环境中 block 尺寸的差异
python train.py task=ShadowHand task.randomize=True headless=False num_envs=64
```

在渲染窗口中,你会看到不同环境里的 Block 尺寸略有不同。

## 高级配置

### 动态随机化(训练过程中随机化)

如果想在训练过程中持续改变 Block 尺寸,需要修改 `setup_only` 参数:

```bash
python train.py task=ShadowHand \
    task.randomize=True \
    task.randomization_params.actor_params.object.scale.setup_only=False \
    task.randomization_params.frequency=720
```

- `setup_only=False`: 允许运行时随机化
- `frequency=720`: 每 720 个仿真步更新一次随机参数

**注意**: 运行时改变尺寸可能导致物理不稳定,建议保持 `setup_only=True`

### 使用不同的分布类型

```bash
# 使用高斯(正态)分布替代均匀分布
python train.py task=ShadowHand \
    task.randomize=True \
    task.randomization_params.actor_params.object.scale.distribution="gaussian"
```

支持的分布类型:
- `"uniform"`: 均匀分布
- `"gaussian"`: 高斯分布
- `"loguniform"`: 对数均匀分布

## 训练效果

启用 Block 尺寸随机化的优势:

1. **提高泛化能力**: 训练的策略可以处理不同尺寸的物体
2. **增强鲁棒性**: 减少对特定尺寸的过拟合
3. **Sim-to-Real**: 更容易迁移到真实机器人(真实物体尺寸可能有误差)

## 常见问题

### Q1: 为什么训练收敛变慢了?

A: 域随机化增加了任务难度,可能需要:
- 增加训练步数
- 调整学习率
- 减小随机化范围

### Q2: 如何只随机化 Block 尺寸,不随机化其他参数?

A: 需要注释掉配置文件中其他不需要的随机化参数,或创建自定义配置文件。

### Q3: 尺寸范围应该设置多大?

A: 建议从小范围开始(如 ±5%),逐步增加:
- 初期训练: `[0.95, 1.05]`
- 中期训练: `[0.90, 1.10]`
- 高级训练: `[0.80, 1.20]`

## 相关文件

- 配置文件: `isaacgymenvs/cfg/task/ShadowHand.yaml`
- 任务实现: `isaacgymenvs/tasks/shadow_hand.py`
- 训练脚本: `isaacgymenvs/train.py`
- 域随机化文档: `docs/domain_randomization.md`

## 参考命令汇总

```bash
# 1. 基础随机化训练
python train.py task=ShadowHand task.randomize=True

# 2. 自定义尺寸范围 (90%-110%)
python train.py task=ShadowHand task.randomize=True \
    task.randomization_params.actor_params.object.scale.range=[0.9,1.1]

# 3. 大范围随机化 (80%-120%)
python train.py task=ShadowHand task.randomize=True \
    task.randomization_params.actor_params.object.scale.range=[0.8,1.2]

# 4. 完整训练配置
python train.py task=ShadowHand \
    task.randomize=True \
    task.randomization_params.actor_params.object.scale.range=[0.9,1.1] \
    num_envs=16384 \
    headless=True \
    train.params.config.max_epochs=10000

# 5. 测试已训练模型
python train.py task=ShadowHand \
    task.randomize=True \
    test=True \
    checkpoint=runs/ShadowHand/nn/your_checkpoint.pth
```

## 总结

要启用 Block 尺寸随机化,最简单的方法就是:

```bash
python train.py task=ShadowHand task.randomize=True
```

这将使用配置文件中预设的所有域随机化参数,包括 Block 尺寸在 95%-105% 之间随机变化。

---

**提示**: 如需更详细的域随机化说明,请查阅 `docs/domain_randomization.md` 文档。
