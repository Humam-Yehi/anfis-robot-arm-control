import matplotlib.pyplot as plt
import numpy as np
import os


def plot_dashboard(q_pid, q_anfis, q_des,
                   tau_pid, tau_anfis,
                   pid_metrics, anfis_metrics,
                   improvement,
                   save_path="results/dashboard.png"):

    time = np.arange(len(q_des))

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))

    # Joint 1 Tracking
    axs[0, 0].plot(time, q_des[:, 0], '--', label="Desired")
    axs[0, 0].plot(time, q_pid[:, 0], label="PID")
    axs[0, 0].plot(time, q_anfis[:, 0], label="ANFIS")
    axs[0, 0].set_title("Joint 1 Tracking")
    axs[0, 0].legend()

    # Joint 2 Tracking
    axs[0, 1].plot(time, q_des[:, 1], '--', label="Desired")
    axs[0, 1].plot(time, q_pid[:, 1], label="PID")
    axs[0, 1].plot(time, q_anfis[:, 1], label="ANFIS")
    axs[0, 1].set_title("Joint 2 Tracking")
    axs[0, 1].legend()

    # Joint 1 Error
    axs[1, 0].plot(time, q_pid[:, 0] - q_des[:, 0], label="PID")
    axs[1, 0].plot(time, q_anfis[:, 0] - q_des[:, 0], label="ANFIS")
    axs[1, 0].set_title("Joint 1 Error")
    axs[1, 0].legend()

    # Joint 2 Error
    axs[1, 1].plot(time, q_pid[:, 1] - q_des[:, 1], label="PID")
    axs[1, 1].plot(time, q_anfis[:, 1] - q_des[:, 1], label="ANFIS")
    axs[1, 1].set_title("Joint 2 Error")
    axs[1, 1].legend()

    # Joint 1 Torque
    axs[2, 0].plot(time, tau_pid[:, 0], label="PID")
    axs[2, 0].plot(time, tau_anfis[:, 0], label="ANFIS")
    axs[2, 0].set_title("Joint 1 Torque")
    axs[2, 0].legend()

    # Joint 2 Torque
    axs[2, 1].plot(time, tau_pid[:, 1], label="PID")
    axs[2, 1].plot(time, tau_anfis[:, 1], label="ANFIS")
    axs[2, 1].set_title("Joint 2 Torque")
    axs[2, 1].legend()

    # Metrics Panel
    text = (
        f"PID\n"
        f"MAE: {pid_metrics['MAE']:.4f}\n"
        f"RMSE: {pid_metrics['RMSE']:.4f}\n\n"
        f"ANFIS\n"
        f"MAE: {anfis_metrics['MAE']:.4f}\n"
        f"RMSE: {anfis_metrics['RMSE']:.4f}\n\n"
        f"Improvement\n"
        f"MAE: {improvement['MAE']:.2f}%\n"
        f"RMSE: {improvement['RMSE']:.2f}%"
    )

    axs[0, 2].axis("off")
    axs[0, 2].text(0.1, 0.5, text)

    axs[1, 2].axis("off")
    axs[2, 2].axis("off")

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path)
    plt.show()
