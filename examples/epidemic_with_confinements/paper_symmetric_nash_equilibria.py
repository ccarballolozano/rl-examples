################################################################################################
# This file is reproduces results from the paper:
# "Efficiency of Symmetric Nash Equilibria in Epidemic Models with Confinements"
# by Maider Sanchez, and Josu Doncel.
# The paper is available at: https://josudoncel.github.io/publicat/2023_Valuetools_SIRC_.pdf
# The original code is available at: https://github.com/josudoncel/StudentsCode/tree/main/MaiderSanchezJimenez
################################################################################################


import argparse

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    N = args.N
    encounter_prob = args.encounter_prob
    recovery_prob = args.recovery_prob
    cost_infection = args.cost_infection
    cost_lockdown = args.cost_lockdown
    discount_factor = args.discount_factor
    theta = args.theta
    max_iter_nash = args.max_iter_nash

    f, axs = plt.subplots(1, 2)

    # Compute global optimum policy
    global_policy, global_V = global_optimum_policy(
        N,
        encounter_prob,
        recovery_prob,
        cost_infection,
        cost_lockdown,
        discount_factor,
        theta,
    )
    # Plot global optimum policy
    for i, j in [(i, j) for i in range(N + 1) for j in range(N + 1) if i + j <= N]:
        if global_policy[i, j] == 0:
            axs[0].plot(i, j, "x", color="red", label="confinement")
        else:
            axs[0].plot(i, j, "o", color="blue", label="max exposure")
    axs[0].set_title("Global Optimum Policy")
    axs[0].set_xlabel("Susceptible")
    axs[0].set_ylabel("Infected")
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys())

    # Compute Nash Equilibrium Policy
    equilibrium_policy = compute_nash_equilibrium(
        N,
        encounter_prob,
        recovery_prob,
        cost_infection,
        cost_lockdown,
        discount_factor,
        theta,
        max_iterations=max_iter_nash,
    )
    # Plot Nash Equilibrium Policy
    for i, j in [(i, j) for i in range(N + 1) for j in range(N + 1) if i + j <= N]:
        if equilibrium_policy[i, j] == 0:
            axs[1].plot(i, j, "x", color="red", label="confinement")
        else:
            axs[1].plot(i, j, "o", color="blue", label="max exposure")
    axs[1].set_title("Nash Equilibrium Policy")
    axs[1].set_xlabel("Susceptible")
    axs[1].set_ylabel("Infected")
    handles, labels = axs[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[1].legend(by_label.values(), by_label.keys())

    plt.show()


def global_optimum_policy(
    N: int,
    encounter_prob: float,
    recovery_prob: float,
    cost_infection: float,
    cost_lockdown: float,
    discount_factor: float,
    theta: float,
):
    k = 0
    V = np.zeros((N + 1, N + 1))
    while True:
        k += 1
        V_k = V.copy()
        V[0, 0] = 0
        for i in range(1, N + 1):
            V[i, 0] = min([i / N * (cost_lockdown - pi) for pi in [0, 1]])
        for j in range(1, N + 1):
            V[0, j] = cost_infection * (j / N) + discount_factor * (
                recovery_prob * V_k[0, j - 1]
            )
        for i, j in [
            (i, j) for i in range(1, N + 1) for j in range(1, N + 1) if i + j <= N
        ]:
            V[i, j] = min(
                [
                    cost_infection * (j / N)
                    + (cost_lockdown - pi) * (i / N)
                    + discount_factor
                    * (
                        recovery_prob * V_k[i, j - 1]
                        + encounter_prob * (j / N) * pi * V_k[i - 1, j + 1]
                    )
                    for pi in [0, 1]
                ]
            )
        delta = np.max(np.abs(V - V_k))
        logger.debug(f"Iteration {k}, Delta: {delta}")
        if delta < theta:
            break

    policy = np.zeros((N + 1, N + 1))
    for i in range(1, N + 1):
        policy[i, 0] = np.argmin([i / N * (cost_lockdown - pi) for pi in [0, 1]])
    for j in range(1, N + 1):
        policy[0, j] = np.argmin(
            [
                cost_infection * (j / N)
                + discount_factor * (recovery_prob * V[0, j - 1])
                for pi in [0, 1]
            ]
        )
    for i, j in [
        (i, j) for i in range(1, N + 1) for j in range(1, N + 1) if i + j <= N
    ]:
        policy[i, j] = np.argmin(
            [
                cost_infection * (j / N)
                + (cost_lockdown - pi) * (i / N)
                + discount_factor
                * (
                    recovery_prob * V[i, j - 1]
                    + encounter_prob * (j / N) * pi * V[i - 1, j + 1]
                )
                for pi in [0, 1]
            ]
        )
    return policy, V


def best_response(
    N: int,
    encounter_prob: float,
    recovery_prob: float,
    cost_infection: float,
    cost_lockdown: float,
    discount_factor: float,
    theta: float,
    pi: np.ndarray,
):
    assert pi.shape == (N + 1, N + 1)
    k = 0
    V = np.zeros((2, N + 1, N + 1))
    while True:
        k += 1
        V_k = V.copy()
        for i in range(N + 1):
            V[0, i, 0] = np.min([cost_lockdown - pi_i for pi_i in [0, 1]])
        for j in range(1, N + 1):
            V[0, 0, j] = np.min(
                [
                    cost_lockdown
                    - pi_i
                    + discount_factor
                    * (
                        encounter_prob * (j / N) * pi_i * V_k[1, 0, j]
                        + recovery_prob * (j / N) * V[0, 0, j - 1]
                    )
                    for pi_i in [0, 1]
                ]
            )
        for i, j in [
            (i, j) for i in range(1, N + 1) for j in range(1, N + 1) if i + j <= N
        ]:
            V[0, i, j] = np.min(
                [
                    cost_infection
                    - pi_i
                    + discount_factor
                    * (
                        encounter_prob * (j / N) * pi_i * V_k[1, i, j]
                        + encounter_prob * (j / N) * pi[i, j] * V_k[1, i - 1, j + 1]
                        + recovery_prob * (j / N) * V_k[0, i, j - 1]
                    )
                    for pi_i in [0, 1]
                ]
            )

        for i in range(N + 1):
            V[1, i, 0] = cost_infection
        for j in range(1, N + 1):
            V[1, 0, j] = (
                cost_infection
                + discount_factor * recovery_prob * (j / N) * V_k[1, 0, j - 1]
            )
        for i, j in [
            (i, j) for i in range(1, N + 1) for j in range(1, N + 1) if i + j <= N
        ]:
            V[1, i, j] = cost_infection + discount_factor * (
                encounter_prob * (j / N) * pi[i, j] * V_k[1, i - 1, j + 1]
                + recovery_prob * (j / N) * V_k[1, i, j - 1]
            )
        delta = np.max(np.abs(V - V_k))
        logger.debug(f"Iteration {k}, Delta: {delta}")
        if delta < theta:
            break

    pi_i = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        pi_i[i, 0] = np.argmin([cost_lockdown - pi_i for pi_i in [0, 1]])
    for j in range(1, N + 1):
        pi_i[0, j] = np.argmin(
            [
                cost_lockdown
                - pi_i
                + discount_factor
                * (
                    encounter_prob * (j / N) * pi_i * V[1, 0, j]
                    + recovery_prob * (j / N) * V[0, 0, j - 1]
                )
                for pi_i in [0, 1]
            ]
        )
    for i, j in [
        (i, j) for i in range(1, N + 1) for j in range(1, N + 1) if i + j <= N
    ]:
        pi_i[i, j] = np.argmin(
            [
                cost_infection
                - pi_i
                + discount_factor
                * (
                    encounter_prob * (j / N) * pi_i * V[1, i, j]
                    + encounter_prob * (j / N) * pi[i, j] * V[1, i - 1, j + 1]
                    + recovery_prob * (j / N) * V[0, i, j - 1]
                )
                for pi_i in [0, 1]
            ]
        )
    return pi_i, V


def compute_nash_equilibrium(
    N: int,
    encounter_prob: float,
    recovery_prob: float,
    cost_infection: float,
    cost_lockdown: float,
    discount_factor: float,
    theta: float,
    max_iterations=None,
):
    logger.info("Compute Nash Equilibrium")
    pi = np.random.randint(0, 1, size=(N + 1, N + 1))
    k = 0
    while True:
        k += 1
        pi_k = pi.copy()
        pi, _ = best_response(
            N,
            encounter_prob,
            recovery_prob,
            cost_infection,
            cost_lockdown,
            discount_factor,
            theta,
            pi_k,
        )
        delta = np.max(np.abs(pi - pi_k))
        logger.debug(f"Iteration {k}, Delta: {delta}")
        if delta < theta:
            logger.info(f"Converged after {k} iterations with delta {delta}")
            break
        elif k is not None and k == max_iterations:
            logger.info(f"Stopped without convergence after {k} iterations")
            break
    return pi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=15, help="Population size")
    parser.add_argument(
        "--encounter_prob", type=float, default=0.6, help="Encounter probability"
    )
    parser.add_argument(
        "--recovery_prob", type=float, default=0.4, help="Recovery probability"
    )
    parser.add_argument(
        "--cost_infection", type=float, default=6, help="Cost of infection"
    )
    parser.add_argument(
        "--cost_lockdown", type=float, default=2, help="Cost of lockdown"
    )
    parser.add_argument(
        "--discount_factor", type=float, default=0.99, help="Discount factor"
    )
    parser.add_argument(
        "--theta", type=float, default=1e-5, help="Threshold for convergence"
    )
    parser.add_argument(
        "--max_iter_nash",
        type=int,
        default=50,
        help="Maximum number of iterations for Nash equilibrium computation",
    )
    args = parser.parse_args()
    logger.info(args)
    main(args)
