import numpy as np


def finetune_controller(arm, model, trajectory_func,
                        duration=5.0, dt=0.01, epochs=20, lr=0.0001):

    for epoch in range(epochs):

        q = np.zeros(2)
        q_dot = np.zeros(2)
        total_error = 0

        steps = int(duration / dt)

        for i in range(steps):
            t = i * dt
            q_des, q_dot_des = trajectory_func(t)

            x = np.concatenate([q, q_dot, q_des, q_dot_des])
            tau = model.forward(x)

            q, q_dot = arm.step(q, q_dot, tau, dt)

            error = q - q_des
            total_error += np.sum(error ** 2)

            firing = model.compute_firing_strengths(x)
            x_ext = np.append(x, 1.0)

            for r in range(model.n_rules):
                grad = np.outer(x_ext, error) * firing[r]
                model.weights[r] -= lr * grad

        print(f"Fine-tune Epoch {epoch+1}/{epochs}, "
              f"Tracking Error: {total_error/steps:.6f}")
