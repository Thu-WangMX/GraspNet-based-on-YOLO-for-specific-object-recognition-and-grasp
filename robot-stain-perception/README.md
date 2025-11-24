# 地面污渍感知系统 (Robot Stain Perception System)

本项目是一个为清洁机器人设计的、基于YOLOv8和Intel RealSense深度相机的实时地面污渍感知系统。它被封装为一个ROS2功能包，能够检测并区分**液体污渍 (liquid)** 和 **固体垃圾 (solid)**，并提供其在三维空间中的位置和尺寸信息，为机器人的自主清洁任务（如拖地、抓取）提供决策依据。

## ✨ 主要功能

- **实时多类别检测**: 利用YOLOv8实时检测视野中的`liquid`和`solid`两类目标。
- **三维空间定位**: 结合RealSense深度相机，输出每个检测目标区域的**真实世界距离**。
- **任务规划决策**: 内置基本逻辑，根据检测到的目标类型，输出建议的清洁任务。
- **标准化接口**: 作为ROS2节点运行，通过标准话题（Topic）发布和订阅数据，易于集成到现有机器人系统中。
- **隐私保护设计**: 纯本地化运算，所有图像数据均在机器人机载计算机上处理，不依赖任何云服务。

## 🔧 系统要求 (Prerequisites)

在部署本系统前，请确保你的机器人平台满足以下软硬件要求。

#### 硬件 (Hardware)

- **机载计算机**: 一台搭载NVIDIA GPU的计算机 (例如: NVIDIA Jetson AGX/Orin/Xavier, 或装有NVIDIA显卡的Intel NUC)。
- **深度相机**: 一台Intel RealSense D400系列深度相机 (例如: D435i, D455)。

#### 软件 (Software)

- **操作系统**: Ubuntu 22.04 LTS
- **机器人框架**: ROS2 Humble Hawksbill
- **NVIDIA环境**:
  - NVIDIA 驱动
  - CUDA Toolkit (建议版本 11.8+)
  - cuDNN
- **基础工具**: Git 和 Git LFS

## 🚀 部署与安装流程

请按照以下步骤在**机器人的机载计算机**上进行部署。

### 1. 安装核心系统依赖

首先，安装Git, Git-LFS, 和ROS2的构建工具。

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs colcon-common-extensions
```

### 2. 设置Git LFS

此步骤只需在你的机器上执行一次，以确保能正确下载模型文件。

```bash
git lfs install
```

### 3. 创建并进入ROS2工作空间

如果你还没有ROS2工作空间，请创建一个。

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### 4. 克隆项目代码

进入 `src` 目录，克隆本项目的代码仓库。

```bash
cd src
# 将下面的URL替换成你自己的Git仓库地址
git clone https://github.com/Hjj04/robot-stain-perception
```
Git LFS会自动下载存储在LFS中的**你自己训练好的模型** (`weights/multiclass_detector_best.pt`)。

### 5. 安装Python依赖环境 (使用Conda)

本项目使用Conda管理Python环境，以隔离依赖。

```bash
# 进入项目目录
cd robot-stain-perception

# 如果你的机器人上没有安装conda，推荐安装Miniconda (Conda的最小安装包)
# wget "[https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh)"  # (适用于Jetson等ARM64架构)
# wget "[https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)" # (适用于Intel/AMD等x86_64架构)
# bash Miniconda3-latest-Linux-*.sh
# source ~/.bashrc

# 使用项目提供的 environment.yml 文件创建并激活conda环境。
# 这个步骤会自动安装 PyTorch, OpenCV, ultralytics (YOLOv8) 等所有必需的Python库。
conda env create -f environment.yml
conda activate stain_env
```

### 6. 安装ROS2相关依赖

包括RealSense的ROS2驱动包和其他必要的ROS消息库。

```bash
# 安装RealSense官方ROS2驱动
sudo apt-get install -y ros-humble-realsense2-camera

# 使用rosdep自动安装package.xml中声明的其他ROS2依赖
cd ~/ros2_ws
rosdep install -i --from-path src --rosdistro humble -y
```

### 7. 编译ROS2工作空间

回到工作空间根目录，使用 `colcon` 进行编译。编译过程会根据`setup.py`文件，将我们的Python脚本注册为ROS2的可执行节点。

```bash
cd ~/ros2_ws
colcon build --packages-select perception_node
```

### 8. 下载YOLOv8预训练权重

我们的模型是在YOLOv8n的官方预训练权重基础上进行微调的。虽然在训练时脚本会自动下载，但在部署时我们最好提前准备好此文件。

```bash
# 进入存放权重的目录
cd ~/ros2_ws/src/robot-stain-perception/weights

# 下载 yolov8n.pt 检测模型权重
wget [https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt)
```

## ▶️ 启动与使用方法

系统被封装为一个ROS2 Launch文件，可以通过一条命令启动所有相关节点。

### 1. 启动感知系统

每次打开新的终端，都需要先加载ROS2工作空间的环境变量和Conda环境。

```bash
# 1. 加载ROS2工作空间环境
source ~/ros2_ws/install/setup.bash

# 2. 激活Conda环境
conda activate stain_env

# 3. 启动Launch文件
#    这条命令会同时启动RealSense相机节点和我们自己的感知节点
ros2 launch perception_node perception.launch.py
```

### 2. 预期输出

启动成功后，你会看到：
* **一个视频窗口**: 标题为 "Robot Task Planner View"，实时显示摄像头画面、检测到的污渍（带颜色框）以及底部的任务决策信息（英文）。
* **终端日志**: ROS2节点会启动并打印初始化信息。当检测到污渍时，`perception_node` 会在终端打印详细的检测日志。

### 3. 查看ROS2话题 (调试)

你可以打开一个新的终端来检查感知节点是否在正常发布数据。

```bash
# 先加载环境
source ~/ros2_ws/install/setup.bash

# 查看当前所有话题，确认相机和感知节点的话题都已启动
ros2 topic list

# 监听感知结果话题，实时查看发布的结构化数据
ros2 topic echo /perception/detected_stains
```

## 🤖 系统架构与ROS2接口

本功能包 (`perception_node`) 通过以下ROS2接口与机器人其他系统交互：

#### 订阅的话题 (Subscriptions)
- `/camera/color/image_raw` (`sensor_msgs/msg/Image`): 原始彩色图像，由`realsense-ros`节点发布。
- `/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/msg/Image`): 与彩色图对齐后的深度图像，由`realsense-ros`节点发布。

#### 发布的话题 (Publications)
- `/perception/detected_stains` (`perception_node/msg/DetectionResult`): 发布检测到的所有污渍的结构化信息，供机器人的决策和规划模块使用。

#### 自定义消息格式 (`msg/`)
- **`DetectionResult.msg`**:
  ```
  std_msgs/Header header
  perception_node/DetectedStain[] stains
  ```
- **`DetectedStain.msg`**:
  ```
  string class_name        # "liquid" 或 "solid"
  float32 confidence     # 置信度
  vision_msgs/BoundingBox2D bbox # 边界框 (将在ROS2节点中转换为此格式)
  float32 depth_median_m # 区域深度中值（米）
  ```

---
**作者**: hjj
**日期**: 2025年10月
