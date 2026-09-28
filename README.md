# 🚗 Autonomous Vehicle Driving System with CARLA Simulator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.x-green.svg)](https://carla.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An AI-powered autonomous driving system that combines computer vision, deep learning, and sensor fusion to navigate vehicles in the CARLA simulator.

---

## 📖 Overview

This project implements a fully autonomous driving system capable of:
- **Lane Detection** using Attention U-Net deep learning model
- **Traffic Sign & Object Detection** with YOLO
- **Real-time Sensor Fusion** (LiDAR, Radar, RGB Camera, Semantic Camera)
- **Intelligent Decision Making** for navigation, speed control, and collision avoidance
- **Route Planning** with automatic waypoint following

### 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🛣️ **Lane Detection** | Semantic segmentation using Attention U-Net for precise lane boundary detection |
| 🚦 **Traffic Light Recognition** | Real-time detection and response to traffic signals |
| 🚧 **Obstacle Detection** | YOLO-based detection of vehicles, pedestrians, and road hazards |
| 🕳️ **Pothole Detection** | Custom trained model for road surface anomaly detection |
| 📡 **Sensor Fusion** | Combines LiDAR, Radar, and Camera data for robust perception |
| 🧠 **PID Control** | Smooth steering and speed control using PID controller |
| 🗺️ **Route Planning** | Automatic path generation using CARLA's Global Route Planner |

---

## 🚀 Quick Start

### Prerequisites

- **CARLA Simulator** (0.9.x) - [Download here](https://github.com/carla-simulator/carla/releases)
- **Python 3.8+**
- **CUDA-capable GPU** (recommended for real-time performance)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autonomous-vehicle-carla.git
cd autonomous-vehicle-carla

# Install dependencies
pip install -r requirements.txt

# Set CARLA PythonAPI path
# Windows: set PYTHONPATH=%PYTHONPATH%;C:\Carla\PythonAPI\carla
# Linux/Mac: export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla
```

### Running the System

1. **Start CARLA Simulator**
   ```bash
   # Windows
   CarlaUE4.exe -RenderOffScreen
   
   # Linux
   ./CarlaUE4.sh -RenderOffScreen
   ```

2. **Launch the Autonomous Driver**
   ```bash
   cd carla
   python lane_and_sign_detection.py
   ```

3. **Controls**
   - The vehicle drives autonomously
   - Press `Q` in the OpenCV window to exit

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SENSOR LAYER                              │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│  RGB Camera │   LiDAR     │   Radar     │  Semantic Camera    │
│  (800x450)  │  (32 ch)    │  (20° FOV)  │  (800x450)          │
└──────┬──────┴──────┬──────┴──────┬──────┴──────────┬──────────┘
       │              │             │                 │
       ▼              ▼             ▼                 ▼
┌─────────────┐ ┌─────────────┐ ┌────────────┐ ┌─────────────────┐
│  YOLO v8    │ │  Distance   │ │  Closing   │ │  Lane/Road      │
│  Detection  │ │  Filtering  │ │  Velocity  │ │  Segmentation   │
│             │ │  (Cone+Height│ │  Detection │ │                 │
│  • Vehicles │ │  +Cluster)  │ │            │ │  • Lane lines   │
│  • Potholes │ │             │ │            │ │  • Road surface │
│  • Signs    │ │             │ │            │ │                 │
└──────┬──────┘ └──────┬──────┘ └──────┬─────┘ └────────┬────────┘
       │              │             │                │
       └──────────────┴──────┬──────┴────────────────┘
                             ▼
              ┌────────────────────────────┐
              │      FUSION & CONTROL       │
              │  ┌──────────────────────┐    │
              │  │   Decision Engine    │    │
              │  │  • Priority-based    │    │
              │  │  • Safety override   │    │
              │  └──────────────────────┘    │
              │  ┌──────────────────────┐      │
              │  │    PID Controller    │      │
              │  │  (Steering + Speed)  │      │
              │  └──────────────────────┘      │
              └────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  CARLA Vehicle │
                    │    Control     │
                    └────────────────┘
```

---

## 🔧 Technical Details

### 1. Attention U-Net for Lane Detection

**File:** [`carla/attention_unet.py`](carla/attention_unet.py)

The Attention U-Net architecture enhances traditional U-Net with attention gates that focus on relevant spatial regions:

```python
# Architecture Overview
Input (3 channels: RGB)
    ↓
Encoder: ConvBlock(3→64) → ConvBlock(64→128) → ConvBlock(128→256) → ConvBlock(256→512)
    ↓
Bottleneck: ConvBlock(512→1024)
    ↓
Decoder + Attention Gates:
    • UpConv(1024→512) + AttentionGate + ConvBlock(1024→512)
    • UpConv(512→256) + AttentionGate + ConvBlock(512→256)
    • UpConv(256→128) + AttentionGate + ConvBlock(256→128)
    • UpConv(128→64) + ConvBlock(128→64)
    ↓
Output: Conv2d(64→3 classes: background, lane, road)
```

**Key Components:**
- **ConvBlock**: Double convolution with BatchNorm and ReLU
- **AttentionGate**: Learnable attention mechanism that suppresses irrelevant regions
- **Skip Connections**: Preserve spatial information from encoder to decoder

### 2. Sensor Fusion Logic

| Sensor | Update Rate | Range | Purpose |
|--------|------------|-------|---------|
| **LiDAR** | 20 Hz | 50m | Obstacle distance, forward collision detection |
| **Radar** | Real-time | 30m | Relative velocity for adaptive cruise control |
| **RGB Camera** | 30 FPS | - | Visual detection (YOLO for objects/signs) |
| **Semantic Camera** | 30 FPS | - | Lane/road segmentation (Class 6=lanes, Class 7=road) |
| **Collision Sensor** | Event | - | Emergency brake trigger |

**Sensor Fusion Priorities (Highest to Lowest):**
1. **Collision Detection** → Immediate stop
2. **Traffic Light (Red/Yellow)** → Stop/Prepare
3. **LiDAR Obstacle** (Forward cone, filtered) → Slow/Stop
4. **Radar Closing Speed** → Adaptive throttle
5. **YOLO Objects** → Vehicle/Pothole response
6. **Lane Invasion** → Corrective steering

### 3. Control System

**PID Controller for Steering:**
```python
steer = KP * error + KI * integral + KD * derivative
# KP=1.0, KI=0.02, KD=0.25
```

**Speed Control:**
- Cruise speed: 35 km/h
- Turn speed reduction: Based on steering angle (0.3→0.6 threshold)
- Pothole mode: Reduce to 15 km/h
- Emergency stop: Full brake (1.0)

### 4. LiDAR Filtering Strategy

To reduce false positives from trees, benches, and curbs:

```python
# 1. Forward cone filter: Only points within ±15° of forward direction
cone = fwd_points[angles < 15°]

# 2. Height filter: Ignore ground (z < -1.0m) and tall objects (z > 2.0m)
fwd = pts[(pts.z > -1.0) & (pts.z < 2.0)]

# 3. Cluster filter: Require minimum 5 points to register as obstacle
if len(cone) < 5:
    ignore_detection()
```

### 5. Route Planning

Uses CARLA's `GlobalRoutePlanner` with 2.0m sampling resolution:
```python
grp = GlobalRoutePlanner(world_map, 2.0)
route = grp.trace_route(start.location, end.location)

# Navigation: Follow waypoints with 3m distance threshold
if distance_to_waypoint < 3m:
    advance_to_next_waypoint()
```

---

## 📁 Project Structure

```
autonomous-vehicle-carla/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── carla/                            # Main application code
│   ├── lane_and_sign_detection.py    # Main driving loop & sensor fusion
│   └── attention_unet.py             # Lane segmentation model architecture
│
└── datasets/                         # Training datasets
    └── pathole/                      # Pothole detection dataset
        └── test/
```

---

## 🧠 Model Training

### Lane Detection Model

The Attention U-Net was trained on CARLA semantic segmentation data:

- **Input**: 800×450 RGB images
- **Output**: 3-class segmentation (background, lane markings, road)
- **Classes**: 
  - 0: Background
  - 1: Lane markings (white/yellow lines)
  - 2: Road surface

### Traffic Sign & Pothole Detection

YOLO model trained to detect:
- Traffic signs (stop, yield, speed limits)
- Vehicles (cars, trucks, motorcycles)
- Potholes and road surface anomalies

---

## ⚙️ Configuration

Key parameters in [`carla/lane_and_sign_detection.py`](carla/lane_and_sign_detection.py:16):

```python
# Model Paths
YOLO_MODEL = "path/to/yolo_model.pt"
LANE_MODEL = "path/to/attention_unet.pth"

# Speed Settings
MAX_SPEED = 35           # km/h cruise speed
LIDAR_SLOW_DIST = 12.0   # m - begin slowing
LIDAR_STOP_DIST = 4.0    # m - emergency brake
RADAR_CLOSE_MS = 8.0     # m/s - closing speed threshold

# PID Gains
KP = 1.0    # Proportional gain
KI = 0.02   # Integral gain
KD = 0.25   # Derivative gain
```

---

## 🛠️ Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: carla` | Add CARLA PythonAPI to PYTHONPATH |
| `CUDA out of memory` | Reduce batch size or use CPU (device='cpu') |
| Vehicle stops on turns | LiDAR may detect barriers; adjust LIDAR_STOP_DIST |
| False obstacle detections | Check LiDAR height and cone filters |
| Vehicle stuck | Stuck recovery activates after 4 seconds |

### Performance Tips

- **GPU Mode**: Set `DEVICE = "cuda"` for real-time inference
- **Synchronous Mode**: Enable in CARLA settings for deterministic behavior
- **Sensor Resolution**: Reduce camera resolution for better FPS

---

## 🔮 Future Improvements

- [ ] Deep reinforcement learning for decision making
- [ ] Multi-vehicle coordination
- [ ] Weather condition adaptation (rain/fog/night)
- [ ] HD Map integration for route optimization
- [ ] Behavioral cloning from human driving data

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [CARLA Simulator](https://carla.org/) - Open-source autonomous driving simulator
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

## 📧 Contact

For questions or contributions, please open an issue or contact:
- **Author**: Tarun PV
- **Project**: Final Year Engineering Project

---

<p align="center">
  <b>⭐ Star this repo if you found it helpful! ⭐</b>
</p>
