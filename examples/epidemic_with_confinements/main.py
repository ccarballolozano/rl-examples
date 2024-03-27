from loguru import logger
import matplotlib.pyplot as plt
import numpy as np


ABSORBING_STATE = (-1, -1)


N = 15
cost_infected = 6
cost_lockdown = 2
delta = 0.99
gamma = 0.6
rho = 0.4
# N = 15
# cost_infected = 6
# cost_lockdown = 0.01
# delta = 0.6
# gamma = 0.6
# rho = 0.4

states = [(i, j) for i in range(N + 1) for j in range(N + 1) if i + j <= N]
states += [ABSORBING_STATE]  # Terminal state where no infected individuals are left
actions = [0, 1]


def transition_prob(
    state: tuple[int, int], action: float, next_state: tuple[int, int]
) -> float:
    """
    Transition probability from state to next_state given action

    Parameters
    ----------
    state : tuple
        (S, I) where S is the number of susceptible individuals and I is the number of infected individuals. If I=0, next state is (-1, -1) and is terminal.
    action : float
        0: no lockdown, 1: lockdown
    next_state : tuple
        (S', I') where S' is the number of susceptible individuals and I' is the number of infected individuals
    """
    assert (
        len(state) == len(next_state) == 2
    ), "state and next_state must be tuples of length 2"
    assert state[0] + state[1] <= N, "S + I must be less than or equal to N"
    assert next_state[0] + next_state[1] <= N, "S' + I' must be less than or equal to N"
    if state[0] == state[1] == 0:  # no S, no I
        if next_state == (-1, -1):
            return 1
        else:
            return 0
    elif state[1] == 0:  # no I
        if next_state == (-1, -1):
            return 1
        else:
            return 0
    elif state[0] == 0:  # no S, I -> I-1
        if next_state == (0, state[1] - 1):
            return rho
        # elif next_state == state:
        #    return 1 - rho
        else:
            return 0
    else:  # S, I -> S, I-1 | S-1, I+1
        if next_state == (state[0], state[1] - 1):
            return rho
        elif next_state == (state[0] - 1, state[1] + 1):
            return gamma * (state[1] / N) * action
        # elif next_state == state:
        #    return 1 - rho - gamma * (state[1] / N) * action
        else:
            return 0


def reward(state, action, next_state) -> float:
    if state == (-1, -1):
        cost = 0
    else:
        cost = cost_infected * (state[1] / N) + (cost_lockdown - action) * (
            state[0] / N
        )
    return -cost


def value_iteration(states, actions, discount_factor=0.95, theta=1e-6):
    n_states = len(states)
    n_actions = len(actions)

    V = np.zeros(n_states)
    policy = np.zeros([n_states, n_actions])
    k = 0
    while True:
        delta = 0
        V_k = (
            V.copy()
        )  # Use the same value function to update during a complete iteration.
        for i_s, s in enumerate(states):
            v = V[i_s]
            V[i_s] = max(
                [
                    sum(
                        [
                            transition_prob(s, a, s_next)
                            * (reward(s, a, s_next) + discount_factor * V_k[i_s_next])
                            for i_s_next, s_next in enumerate(states)
                        ]
                    )
                    for i_a, a in enumerate(actions)
                ]
            )
            delta = max(delta, abs(v - V[i_s]))
        k += 1
        logger.debug(f"Iteration {k}, Delta: {delta}")
        v_matrix = np.zeros((N + 1, N + 1))
        # fill v_matrix
        for i, s in enumerate(states):
            v_matrix[s[0], s[1]] = V[i]
        with np.printoptions(
            precision=4,
            suppress=True,
            formatter={"float": "{:0.4f}".format},
            linewidth=200,
        ):
            print(v_matrix)
        # logger.debug(f"Iteration {k}, Delta: {delta}, Value function: {V}")
        if delta < theta:
            break
    for i_s, s in enumerate(states):
        a_best = np.argmax(
            [
                sum(
                    [
                        transition_prob(s, a, next_s)
                        * (reward(s, a, next_s) + discount_factor * V[i_next_s])
                        for i_next_s, next_s in enumerate(states)
                    ]
                )
                for i_a, a in enumerate(actions)
            ]
        )
        vals = [
            sum(
                [
                    transition_prob(s, a, next_s)
                    * (reward(s, a, next_s) + discount_factor * V[i_next_s])
                    for i_next_s, next_s in enumerate(states)
                ]
            )
            for i_a, a in enumerate(actions)
        ]
        # if vals[0] == vals[1]:
        # logger.debug(f"Equal values for state {s}")
        policy[i_s, a_best] = 1
    return policy, V


def main():
    policy, V = value_iteration(states, actions, discount_factor=delta, theta=1e-6)
    print(policy)

    for i_s, s in enumerate(states):
        i_a_best = np.argmax(policy[i_s])
        if i_a_best == 0:
            plt.plot(s[0], s[1], "x", color="red", label="a")
        else:
            plt.plot(s[0], s[1], "o", color="blue")
    plt.show()
    print(0)
    # states = [(i, j) for i in range(N + 1) for j in range(N + 1) if i + j <= N]
    # actions = [0, 1]


#
# n_states = len(states)
# n_actions = len(actions)
#
## Transition probabilities
# transition_probabilities = np.zeros((n_states, n_actions, n_states))
# rewards = np.zeros((n_states, n_actions, n_states))
# for j in range(1, N + 1):
#    transition_probabilities[states.index((0, j)), :, states.index((0, j - 1))] = (
#        rho  # no S, I -> R
#    )
# for i in range(1, N + 1):
#    for j in range(1, N + 1 - i):
#        transition_probabilities[
#            states.index((i, j)), :, states.index((i, j - 1))
#        ] = rho
#
#        transition_probabilities[
#            states.index((i, j)), 0, states.index((i - 1, j + 1))
#        ] = (gamma * (j / N) * 0)
#        transition_probabilities[
#            states.index((i, j)), 1, states.index((i - 1, j + 1))
#        ] = (gamma * (j / N) * 1)
# return


if __name__ == "__main__":
    main()
