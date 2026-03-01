import numpy as np
from models.arm_dynamics import TwoDOFArm
from models.pid import PIDController
from models.anfis import ANFISController
from training.data_generation import generate_dataset
from training.supervised_training import train_supervised
from training.finetuning import finetune_controller
from evaluation.simulation import simulate_controller
from evaluation.metrics import compute_joint_metrics, compute_improvement
from evaluation.dashboard import plot_dashboard


def trajectory(t):
    q_des = np.array([np.sin(t), np.cos(t)])
    q_dot_des = np.array([np.cos(t), -np.sin(t)])
    return q_des, q_dot_des


def run_experiment():

    arm = TwoDOFArm()

    # Generate data
    X, Y = generate_dataset(arm, trajectory)

    # Train ANFIS
    anfis = ANFISController()
    train_supervised(anfis, X, Y)
    finetune_controller(arm, anfis, trajectory)

    # PID baseline
    pid = PIDController(Kp=[50, 50], Kd=[10, 10])

    # Wrap PID into forward interface
    class PIDWrapper:
        def __init__(self, pid):
            self.pid = pid
            self.q = np.zeros(2)
            self.q_dot = np.zeros(2)

        def forward(self, x):
            q = x[0:2]
            q_dot = x[2:4]
            q_des = x[4:6]
            q_dot_des = x[6:8]
            return self.pid.compute(q, q_dot, q_des, q_dot_des, 0.01)

    pid_controller = PIDWrapper(pid)

    # Simulate
    q_pid, q_des, tau_pid = simulate_controller(
        arm, pid_controller, trajectory)

    q_anfis, _, tau_anfis = simulate_controller(
        arm, anfis, trajectory)

    # Metrics
    pid_metrics = compute_joint_metrics(q_pid, q_des)
    anfis_metrics = compute_joint_metrics(q_anfis, q_des)
    improvement = compute_improvement(pid_metrics, anfis_metrics)

    # Dashboard
    plot_dashboard(
        q_pid, q_anfis, q_des,
        tau_pid, tau_anfis,
        pid_metrics, anfis_metrics,
        improvement
    )
