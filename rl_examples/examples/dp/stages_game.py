# https://towardsdatascience.com/getting-started-with-markov-decision-processes-reinforcement-learning-ada7b4572ffb

from loguru import logger

from rl.mdp import MDP
from rl.dp import value_iteration


def main():
    states = [0, 1, 2, 3, 4]
    # 0: Stage1, 1: Stage2, 2: Pause, 3: Win, 4: Stop
    actions = [0, 1, 2, 3, 4, 5, 6] 
    # 0: Chores, 1: Continue, 2: Advance1, 3: Advance2, 4: Teleport, 5: Stop, 6: Complete

    rewards = [
        (0, 2, 1, -2),  # s: stage1, a: advance1, s': stage2, r: -2
        (0, 0, 2, -1),  # s: stage1, a: chores, s': pause, r: -1
        (1, 4, 0, -1),  # s: stage2, a: teleport, s': stage1, r: -1
        (1, 4, 1, 1),   # s: stage2, a: teleport, s': stage2, r: 1
        (1, 3, 3, -2),  # s: stage2, a: advance2, s': win, r: -2
        (1, 5, 4, 0),   # s: stage2, a: stop, s': stop, r: 0
        (2, 0, 2, -1),  # s: pause, a: chores, s': pause, r: -1
        (2, 1, 0, -1),  # s: pause, a: continue, s': stage1, r: -1
        (3, 6, 4, 10)   # s: win, a: continue, s': stop, r: 10
    ]

    transitions = [
        (0, 2, 1, 1),  # s: stage1, a: advance1, s': stage2, prob
        (0, 0, 2, 1),  # s: stage1, a: chores, s': pause, prob
        (1, 4, 0, 0.6),  # s: stage2, a: teleport, s': stage1, prob
        (1, 4, 1, 0.4),  # s: stage2, a: teleport, s': stage2, prob
        (1, 3, 3, 1),  # s: stage2, a: advance2, s': win, prob
        (1, 5, 4, 1),  # s: stage2, a: stop, s': stop, prob
        (2, 0, 2, 1),  # s: pause, a: chores, s': pause, prob
        (2, 1, 0, 1),  # s: pause, a: continue, s': stage1, prob
        (3, 6, 4, 1)  # s: win, a: continue, s': stop, prob
    ]
    mdp = MDP(
        states=states,
        actions=actions,
        transitions=transitions,
        rewards=rewards)
    
    policy, V = value_iteration(mdp, discount_factor=1, theta=1e-8)
    logger.info(f"Value function: {V}")
    logger.info(f"Optimal policy: {policy}")

if __name__ == "__main__":
    main()