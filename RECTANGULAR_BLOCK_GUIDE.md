# Shadow Hand 长方体物体尺寸随机化指南

## 概述

本指南说明如何将 Shadow Hand 任务中的正方体物体改为长方体,并实现长、宽、高三个维度的独立随机化。

## 实现方案

有两种方案可以实现长方体的长宽高随机化:

### 方案 1: 创建自定义长方体 URDF 文件(推荐)

这种方法最简单,不需要修改 Python 代码。

#### 步骤 1: 创建长方体 URDF 文件

在 `assets/urdf/objects/` 目录下创建新文件 `rectangular_box.urdf`:

```bash
cd /home/zhonghao/learning/IsaacGym/IsaacGymEnvs/assets/urdf/objects/
```

创建文件内容:

```xml
<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <!-- 长方体尺寸: 长(x)=0.06, 宽(y)=0.04, 高(z)=0.08 -->
        <box size="0.06 0.04 0.08"/>
      </geometry>
      <material name="multicolor">
        <color rgba="1.0 0.5 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <!-- 碰撞体与视觉体相同 -->
        <box size="0.06 0.04 0.08"/>
      </geometry>
    </collision>
    <inertial>
      <!-- 密度单位: kg/m³ -->
      <density value="567.0"/>
    </inertial>
  </link>
</robot>
```

**尺寸说明**:
- `size="x y z"` 格式,单位为米
- 示例中: 长=6cm, 宽=4cm, 高=8cm
- 可以根据需要调整这些值

#### 步骤 2: 修改配置文件

编辑 `isaacgymenvs/cfg/task/ShadowHand.yaml`,修改 asset 配置:

```yaml
asset:
  assetFileName: mjcf/open_ai_assets/hand/shadow_hand.xml
  assetFileNameBlock: urdf/objects/rectangular_box.urdf  # 改用新的长方体文件
  assetFileNameEgg: mjcf/open_ai_assets/hand/egg.xml
  assetFileNamePen: mjcf/open_ai_assets/hand/pen.xml
```

#### 步骤 3: 配置随机化

在 `ShadowHand.yaml` 中配置长方体的随机化范围:

```yaml
task:
  randomize: True
  randomization_params:
    # ... 其他配置 ...
    actor_params:
      object:
        scale:
          # 整体缩放: 80%-120%
          range: [0.8, 1.2]
          operation: "scaling"
          distribution: "uniform"
          setup_only: True
```

**注意**: 使用 `scale` 参数会等比例缩放长、宽、高。如果想让三个维度独立变化,需要使用方案 2。

#### 步骤 4: 启动训练

```bash
cd /home/zhonghao/learning/IsaacGym/IsaacGymEnvs/isaacgymenvs
python train.py task=ShadowHand num_envs=64 \
    task.randomize=True \
    train.params.config.minibatch_size=256
```

---

### 方案 2: 代码级实现非均匀缩放(高级)

如果想让长、宽、高**独立随机化**(例如:长度变化 80%-120%,宽度变化 60%-80%,高度变化 100%-140%),需要修改 Python 代码。

#### 步骤 1: 创建基础长方体 URDF

同方案 1 的步骤 1,创建 `rectangular_box.urdf` 文件。

#### 步骤 2: 修改 shadow_hand.py

在 `isaacgymenvs/tasks/shadow_hand.py` 中添加非均匀缩放支持。

找到 `_create_envs` 方法中加载物体的部分(约 300 行),在加载物体后添加代码:

```python
# 在 object_asset = self.gym.load_asset(...) 之后添加

# 如果启用了随机化,应用非均匀缩放
if self.randomize and self.object_type == "block":
    # 为每个环境生成随机缩放因子
    self.object_scale_x = torch_rand_float(0.8, 1.2, (self.num_envs, 1), device=self.device)
    self.object_scale_y = torch_rand_float(0.6, 0.8, (self.num_envs, 1), device=self.device)
    self.object_scale_z = torch_rand_float(1.0, 1.4, (self.num_envs, 1), device=self.device)
```

然后在创建每个环境的物体时应用缩放:

```python
# 在环境创建循环中
for i in range(self.num_envs):
    # ... 创建环境代码 ...
    
    # 创建物体
    object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
    
    # 应用非均匀缩放
    if self.randomize and self.object_type == "block":
        object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
        # 注意: 具体的缩放实现需要使用 PhysX 的形状修改 API
        # 这部分较为复杂,建议使用方案 1 或方案 3
```

**注意**: PhysX 在运行时修改几何形状比较复杂,这种方法实现难度较高。

---

### 方案 3: 创建多个预定义尺寸的 URDF(最简单的非均匀方案)

创建多个不同长宽高比例的长方体 URDF 文件,让系统随机选择。

#### 步骤 1: 创建多个 URDF 文件

创建 5 个不同尺寸的长方体文件:

**rectangular_box_1.urdf** (细长型):
```xml
<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.08 0.03 0.05"/>
      </geometry>
      <material name="color1">
        <color rgba="1.0 0.3 0.3 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.08 0.03 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <density value="567.0"/>
    </inertial>
  </link>
</robot>
```

**rectangular_box_2.urdf** (扁平型):
```xml
<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.06 0.06 0.03"/>
      </geometry>
      <material name="color2">
        <color rgba="0.3 1.0 0.3 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.06 0.06 0.03"/>
      </geometry>
    </collision>
    <inertial>
      <density value="567.0"/>
    </inertial>
  </link>
</robot>
```

**rectangular_box_3.urdf** (高瘦型):
```xml
<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.04 0.04 0.09"/>
      </geometry>
      <material name="color3">
        <color rgba="0.3 0.3 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.04 0.04 0.09"/>
      </geometry>
    </collision>
    <inertial>
      <density value="567.0"/>
    </inertial>
  </link>
</robot>
```

**rectangular_box_4.urdf** (均衡型):
```xml
<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.06"/>
      </geometry>
      <material name="color4">
        <color rgba="1.0 1.0 0.3 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.06"/>
      </geometry>
    </collision>
    <inertial>
      <density value="567.0"/>
    </inertial>
  </link>
</robot>
```

**rectangular_box_5.urdf** (宽胖型):
```xml
<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.04 0.07 0.05"/>
      </geometry>
      <material name="color5">
        <color rgba="1.0 0.5 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.04 0.07 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <density value="567.0"/>
    </inertial>
  </link>
</robot>
```

#### 步骤 2: 修改 shadow_hand.py 以随机选择

在 `shadow_hand.py` 的 `__init__` 方法中修改:

```python
# 在 asset_files_dict 定义之后添加
if "asset" in self.cfg["env"]:
    # 如果启用随机化,准备多个长方体文件
    if self.cfg["task"]["randomize"] and self.object_type == "block":
        import random
        box_files = [
            "urdf/objects/rectangular_box_1.urdf",
            "urdf/objects/rectangular_box_2.urdf",
            "urdf/objects/rectangular_box_3.urdf",
            "urdf/objects/rectangular_box_4.urdf",
            "urdf/objects/rectangular_box_5.urdf",
        ]
        # 为每个环境随机选择一个
        self.object_urdf_files = box_files
    else:
        self.asset_files_dict["block"] = self.cfg["env"]["asset"].get(
            "assetFileNameBlock", 
            self.asset_files_dict["block"]
        )
```

然后在 `_create_envs` 方法中修改加载逻辑:

```python
# 在加载 object_asset 时
if hasattr(self, 'object_urdf_files'):
    # 为每个环境随机选择一个文件
    import random
    object_asset_file = random.choice(self.object_urdf_files)
else:
    object_asset_file = self.asset_files_dict[self.object_type]
```

**注意**: 这个方案需要修改代码,但实现相对简单。

---

## 推荐方案总结

### 如果你想要:

1. **简单的整体缩放** (长宽高等比例变化)
   - 使用**方案 1**
   - 只需创建一个 URDF 文件
   - 通过 `scale` 参数控制

2. **多种不同形状的长方体** (预定义几种形状)
   - 使用**方案 3**
   - 创建多个 URDF 文件
   - 需要小幅修改 Python 代码

3. **完全自由的长宽高独立随机化**
   - 使用**方案 2**
   - 需要深入修改代码
   - 实现较复杂

## 快速开始 - 方案 1 示例

### 1. 创建长方体 URDF

```bash
cd /home/zhonghao/learning/IsaacGym/IsaacGymEnvs/assets/urdf/objects/
cat > rectangular_box.urdf << 'EOF'
<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.06 0.04 0.08"/>
      </geometry>
      <material name="orange">
        <color rgba="1.0 0.5 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.06 0.04 0.08"/>
      </geometry>
    </collision>
    <inertial>
      <density value="567.0"/>
    </inertial>
  </link>
</robot>
EOF
```

### 2. 训练命令

```bash
cd /home/zhonghao/learning/IsaacGym/IsaacGymEnvs/isaacgymenvs

# 使用新的长方体,带整体尺寸随机化
python train.py task=ShadowHand \
    num_envs=64 \
    task.randomize=True \
    task.env.asset.assetFileNameBlock=urdf/objects/rectangular_box.urdf \
    task.randomization_params.actor_params.object.scale.range=[0.8,1.2] \
    train.params.config.minibatch_size=256
```

### 3. 测试不同尺寸范围

```bash
# 小范围变化 (90%-110%)
python train.py task=ShadowHand num_envs=64 \
    task.randomize=True \
    task.env.asset.assetFileNameBlock=urdf/objects/rectangular_box.urdf \
    task.randomization_params.actor_params.object.scale.range=[0.9,1.1] \
    train.params.config.minibatch_size=256

# 大范围变化 (70%-130%)
python train.py task=ShadowHand num_envs=64 \
    task.randomize=True \
    task.env.asset.assetFileNameBlock=urdf/objects/rectangular_box.urdf \
    task.randomization_params.actor_params.object.scale.range=[0.7,1.3] \
    train.params.config.minibatch_size=256
```

## 可视化验证

启用可视化模式查看长方体效果:

```bash
python train.py task=ShadowHand num_envs=16 \
    task.randomize=True \
    task.env.asset.assetFileNameBlock=urdf/objects/rectangular_box.urdf \
    headless=False \
    train.params.config.minibatch_size=128
```

在渲染窗口中你会看到:
- 长方体形状(不再是正方体)
- 不同环境中的尺寸差异(如果启用了随机化)

## 尺寸选择建议

### 长方体基础尺寸参考:

- **小型**: `0.04 0.03 0.05` (4cm × 3cm × 5cm)
- **中型**: `0.06 0.04 0.08` (6cm × 4cm × 8cm) - **推荐**
- **大型**: `0.08 0.05 0.10` (8cm × 5cm × 10cm)

### 随机化范围建议:

- **保守**: `[0.90, 1.10]` (±10%) - 适合初期训练
- **中等**: `[0.80, 1.20]` (±20%) - 推荐值
- **激进**: `[0.70, 1.30]` (±30%) - 高难度

## 常见问题

### Q1: 长方体会不会比正方体更难抓取?

A: 是的,尤其是细长的长方体。建议:
- 从接近正方体的比例开始 (如 5:4:6)
- 逐步增加长宽高的差异
- 调整奖励参数以适应新的形状

### Q2: 如何调整长方体的初始姿态?

A: 修改 `object_start_pose` 的旋转:

```python
object_start_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 1.57)  # 旋转90度
```

### Q3: 长方体质心会影响物理行为吗?

A: 会的。URDF 中的 `<inertial>` 标签使用 `density` 自动计算惯性。如果需要手动指定:

```xml
<inertial>
  <mass value="0.1"/>
  <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
</inertial>
```

## 下一步

1. 创建长方体 URDF 文件
2. 测试基础训练是否正常
3. 逐步增加随机化范围
4. 根据训练效果调整奖励函数

---

**提示**: 建议从方案 1 开始,这是最简单且最稳定的方法!
