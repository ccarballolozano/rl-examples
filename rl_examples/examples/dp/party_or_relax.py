# https://artint.info/3e/html/ArtInt3e.Ch12.S5.html - Example 12.29

from loguru import logger

from rl.dp import value_iteration
from rl.mdp import MDP


def main():
    states = [0, 1]  # 0: healthy, 1: sick
    actions = [0, 1]  # 0: relax, 1: party

    transitions = [
        (0, 0, 0, 0.95),  # s: healthy, a: relax, s': healthy, prob
        (0, 0, 1, 0.05),  # s: healthy, a: relax, s': sick, prob
        (0, 1, 0, 0.7),  # s: healthy, a: party, s': healthy, prob
        (0, 1, 1, 0.3),  # s: healthy, a: party, s': sick, prob
        (1, 0, 0, 0.5),  # s: sick, a: relax, s': healthy, prob
        (1, 0, 1, 0.5),  # s: sick, a: relax, s': sick, prob
        (1, 1, 0, 0.1),  # s: sick, a: party, s': healthy, prob
        (1, 1, 1, 0.9),  # s: sick, a: party, s': sick, prob
    ]

    rewards = [
        (0, 0, 0, 7),  # s: healthy, a: relax, s': healthy, r: 7
        (0, 0, 1, 7),  # s: healthy, a: relax, s': sick, r: 7
        (0, 1, 0, 10),  # s: healthy, a: party, s': healthy, r: 10
        (0, 1, 1, 10),  # s: healthy, a: party, s': sick, r: 10
        (1, 0, 0, 0),  # s: sick, a: relax, s': healthy, r: 0
        (1, 0, 1, 0),  # s: sick, a: relax, s': sick, r: 0
        (1, 1, 0, 2),  # s: sick, a: party, s': healthy, r: 2
        (1, 1, 1, 2),  # s: sick, a: party, s': sick, r: 2
    ]

    mdp = MDP(
        states=states,
        actions=actions,
        transitions=transitions,
        rewards=rewards)

    policy, V = value_iteration(mdp, discount_factor=0.8, theta=1e-8)
    logger.info(f"Value function: {V}")
    logger.info(f"Optimal policy: {policy}")


if __name__ == "__main__":
    main()
