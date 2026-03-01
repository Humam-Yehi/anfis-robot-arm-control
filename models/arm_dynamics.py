import numpy as np


class TwoDOFArm:
    """
    Simple 2-DOF planar robotic arm dynamics.
    """

    def __init__(self):
        self.m1 = 1.0
        self.m2 = 1.0
        self.l1 = 1.0
        self.l2 = 1.0
        self.g = 9.81

    def dynamics(self, q, q_dot, tau):
        q1, q2 = q
        dq1, dq2 = q_dot
        tau1, tau2 = tau

        M = np.array([
            [2 + np.cos(q2), 1 + 0.5 * np.cos(q2)],
            [1 + 0.5 * np.cos(q2), 1]
        ])

        C = np.array([
            [-0.5 * np.sin(q2) * dq2, -0.5 * np.sin(q2) * (dq1 + dq2)],
            [0.5 * np.sin(q2) * dq1, 0]
        ])

        G = np.array([
            self.g * (2 * np.cos(q1) + np.cos(q1 + q2)),
            self.g * np.cos(q1 + q2)
        ])

        q_ddot = np.linalg.inv(M) @ (tau - C @ q_dot - G)
        return q_ddot

    def step(self, q, q_dot, tau, dt):
        q_ddot = self.dynamics(q, q_dot, tau)
        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt
        return q, q_dot
