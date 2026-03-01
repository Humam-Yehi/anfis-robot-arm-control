# ANFIS-Based Intelligent Control for a 2-DOF Robotic Arm

## 📌 Project Overview

This project implements an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** controller for a 2-DOF planar robotic manipulator and compares its performance with a classical PID controller.

The objective is to investigate how neuro-fuzzy learning can improve trajectory tracking performance in robotic arm control.

The framework includes:

- Full dynamic simulation of a 2-DOF robotic arm
- PID baseline controller
- Supervised learning (PID imitation)
- Rollout-based fine-tuning
- Quantitative performance evaluation
- Multi-panel experiment dashboard visualization

---

## 🎯 Motivation

Classical PID controllers are simple and effective but may struggle with nonlinear dynamics and changing operating conditions.

ANFIS combines:
- Fuzzy logic reasoning
- Neural network learning
- Nonlinear function approximation

This makes it a strong candidate for intelligent robotic control.


## ⚙️ System Description

### 2-DOF Robotic Arm

- Planar manipulator
- Nonlinear dynamic model
- Gravity, Coriolis, and inertia effects included
- Numerical integration using discrete time stepping

### Controllers

#### PID Controller
- Classical joint-space controller
- Used to generate expert demonstrations
- Serves as performance baseline

#### ANFIS Controller
- Multi-rule fuzzy inference system
- Gaussian membership functions
- Linear consequent parameters
- Two-output torque prediction
- Trained via:
  1. Supervised imitation learning
  2. Rollout-based fine-tuning

---

## 🧠 Methodology

1. Generate trajectory (sinusoidal reference)
2. Use PID to generate training dataset
3. Train ANFIS using supervised learning
4. Fine-tune using closed-loop tracking error
5. Compare PID vs ANFIS performance
6. Visualize results in a dashboard

---

## 📊 Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Maximum absolute error
- Percentage improvement over PID

---

## ▶️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/anfis-robot-arm-control.git
cd anfis-robot-arm-control
2️⃣ Install dependencies
pip install numpy matplotlib
3️⃣ Run the experiment
python main.py

