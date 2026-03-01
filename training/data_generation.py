import numpy as np
from models.pid import PIDController


def generate_dataset(arm, trajectory_func, duration=5.0, dt=0.01):
    pid = PIDController(Kp=[50, 50], Kd=[10, 10])

    q = np.zeros(2)
    q_dot = np.zeros(2)

    X, Y = [], []

    steps = int(duration / dt)

    for i in range(steps):
        t = i * dt
        q_des, q_dot_des = trajectory_func(t)

        tau = pid.compute(q, q_dot, q_des, q_dot_des, dt)

        x = np.concatenate([q, q_dot, q_des, q_dot_des])
        X.append(x)
        Y.append(tau)

        q, q_dot = arm.step(q, q_dot, tau, dt)

    return np.array(X), np.array(Y)
