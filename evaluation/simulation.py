import numpy as np


def simulate_controller(arm, controller, trajectory_func,
                        duration=5.0, dt=0.01):

    q = np.zeros(2)
    q_dot = np.zeros(2)

    qs = []
    q_des_list = []
    taus = []

    steps = int(duration / dt)

    for i in range(steps):
        t = i * dt
        q_des, q_dot_des = trajectory_func(t)

        x = np.concatenate([q, q_dot, q_des, q_dot_des])
        tau = controller.forward(x)

        q, q_dot = arm.step(q, q_dot, tau, dt)

        qs.append(q.copy())
        q_des_list.append(q_des.copy())
        taus.append(tau.copy())

    return (
        np.array(qs),
        np.array(q_des_list),
        np.array(taus)
    )
