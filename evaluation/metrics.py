import numpy as np


def compute_joint_metrics(q, q_des):
    error = q - q_des
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    max_error = np.max(np.abs(error))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAX": max_error
    }


def compute_improvement(pid_metrics, anfis_metrics):
    improvement = {}
    for key in pid_metrics:
        improvement[key] = (
            (pid_metrics[key] - anfis_metrics[key]) /
            (pid_metrics[key] + 1e-8)
        ) * 100.0
    return improvement
