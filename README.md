# Vision & Spatial Understanding

A computer vision and spatial reasoning project that connects object perception with robotic manipulation in simulation and pedestrian-awareness analysis on real-world street footage.

The project explores how visual information can be transformed into meaningful spatial measurements, motion analysis, and geometry-aware actions.

## 🎬 Project Demos

### 1. Franka Panda — Vision-Based Robotic Manipulation

A vision-to-manipulation pipeline using YOLO-based object perception, camera geometry, 3D localization, and inverse kinematics in MuJoCo.

**[▶ Watch the Robot Manipulation Demo on LinkedIn](https://lnkd.in/p/ga9nV8KJ)**

### 2. Real-World Pedestrian Awareness

A computer vision system that tracks pedestrians and analyses spatial relationships, relative movement, and potential path conflicts in recorded street-walking footage.

**[▶ Watch the Pedestrian Awareness Demo on LinkedIn](https://lnkd.in/p/getdN5Fd)**

---

## 🧠 Project Overview

The project explores two complementary applications of vision-based spatial understanding.

- **Robotic manipulation:** Translating camera observations into world-coordinate targets for robot approach planning and manipulation in a MuJoCo simulation.
- **Pedestrian awareness:** Analysing tracked pedestrians, estimated distances, movement patterns, and potential path conflicts in real-world video.

Both workflows focus on going beyond object detection to understand where objects or people are located and how their spatial relationships change over time.

## 🤖 Part 1 — Vision-Based Robotic Manipulation

### Key Features

- **Object Detection & Segmentation:** Identifying objects from a robot-mounted camera using YOLO-based vision.
- **Object Tracking:** Maintaining object-level tracking identities across frames.
- **3D Localization:** Converting image-space detections into world coordinates using camera geometry, camera pose, and ray–plane intersection.
- **Geometry-Aware Approach Planning:** Computing approach targets relative to detected objects.
- **Inverse Kinematics:** Using Jacobian-based damped least-squares optimization to solve for robot joint configurations.
- **Manipulation Workflow:** Connecting perception, localization, approach, grasping, and object movement within a simulated environment.

### Perception-to-Manipulation Pipeline

```text
Robot-Mounted Camera
        ↓
Object Detection & Segmentation
        ↓
Object Tracking
        ↓
Pixel-to-World Localization
        ↓
3D Target Estimation
        ↓
Geometry-Aware Approach Planning
        ↓
Damped Least-Squares Inverse Kinematics
        ↓
Robot Manipulation in MuJoCo
```

### Localization & IK Evaluation

The workflow was evaluated against MuJoCo simulation ground-truth positions.

| Metric | Observed Result |
|---|---:|
| Object localization error | ~1–4 mm |
| Robot approach-position error | ~0.4–3 mm |

These values represent observed results from the implemented simulation workflow, rather than a formal statistical benchmark across a large evaluation set.

## 🚶 Part 2 — Real-World Pedestrian Awareness

The second workflow applies computer vision and spatial analysis to recorded street-walking footage. It tracks pedestrians and analyses their relative positions and movement to identify potential path conflicts.

### Key Features

- **Pedestrian Detection & Tracking:** Detecting pedestrians and maintaining track IDs across frames.
- **Distance Estimation:** Estimating pedestrian distances using configured camera parameters.
- **Relative Movement Analysis:** Analysing whether tracked pedestrians are approaching or receding.
- **Inter-Person Distance Analysis:** Measuring estimated spatial separation between tracked individuals.
- **Path-Conflict Detection:** Identifying pedestrians whose movement may intersect a configured path.
- **Closest-Approach Prediction:** Estimating time to closest approach and miss distance from observed motion.
- **State Overview:** Displaying a summarized awareness state, including `NORMAL`, `AWARE`, `CAUTION`, and `DANGER`.

### Analysis Pipeline

```text
Recorded Street Video
        ↓
Pedestrian Detection & Tracking
        ↓
Track IDs and Position Estimation
        ↓
Spatial and Relative-Motion Analysis
        ↓
Path-Conflict Assessment
        ↓
Closest-Approach Time & Miss-Distance Estimation
        ↓
State Overview and Visualization
```

### Observed Demo Performance

| Metric | Observed Result |
|---|---:|
| Pedestrians tracked in demonstrated clips | Up to 6 |
| Processing frame rate in sampled clips | ~15–23 FPS |

These figures are observations from demonstrated clips, not standardized benchmark results. Distance estimates and predicted motion depend on camera configuration and the assumptions used by the analysis.

**Current scope:** The pedestrian-awareness workflow analyses recorded video and visualizes potential conflicts. It does not currently provide autonomous robot control or autonomous path rerouting.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Computer Vision | OpenCV, YOLO, instance segmentation, object tracking |
| Deep Learning | PyTorch |
| Spatial Understanding | Camera geometry, camera pose, ray–plane intersection |
| Robotics | MuJoCo, Franka Panda |
| Robot Kinematics | Jacobian-based damped least-squares inverse kinematics |
| Data Analysis | NumPy |
| Visualization | OpenCV-based visualizations and state overlays |

## 🎯 Engineering Focus

This project focuses on the transition from visual perception to spatial understanding.

Key engineering challenges include:

- Mapping image-space observations to meaningful world coordinates.
- Improving localization accuracy through camera geometry and coordinate transformations.
- Solving inverse kinematics for geometry-aware robot approach targets.
- Maintaining pedestrian identities and analysing movement across video frames.
- Estimating spatial relationships and potential path conflicts from visual observations.
- Making perception and analysis outputs interpretable through visual feedback.

## 🔬 Future Work

Potential directions for further development include:

- More extensive quantitative evaluation across different scenes and camera configurations.
- Robustness testing under occlusion, changing viewpoints, and varying lighting.
- Improved temporal motion prediction and uncertainty estimation.
- Integration of higher-level decision-making and planning components.
- Further evaluation of perception accuracy, latency, and end-to-end system performance.

These are future development directions, not claims about currently implemented functionality.

## 👨‍💻 Author

**Mihir Vyas**

Interested in computer vision, machine learning, spatial understanding, robotics, and intelligent perception systems.

**GitHub:** [Innovatewithapple](https://github.com/Innovatewithapple)

---

*This is an experimental engineering project focused on vision-based spatial understanding, robotic manipulation in simulation, and pedestrian-motion analysis.*
