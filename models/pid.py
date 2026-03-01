import numpy as np


class PIDController:
    def __init__(self, Kp, Kd, Ki=None):
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)
        self.Ki = np.array(Ki) if Ki is not None else np.zeros(2)
        self.integral = np.zeros(2)

    def reset(self):
        self.integral = np.zeros(2)

    def compute(self, q, q_dot, q_des, q_dot_des, dt):
        error = q_des - q
        derror = q_dot_des - q_dot
        self.integral += error * dt

        tau = (
            self.Kp * error +
            self.Kd * derror +
            self.Ki * self.integral
        )
        return tau
