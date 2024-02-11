# rl-examples

Reinforcement Learning (RL) Examples

## Run Examples

1. Create the virtual environment with required packages, specified in `dependencies.yaml` file. For example, using conda:

    ```bash
    conda env create -f dependencies.yaml
    ```

2. Activate the virtual environment:

    ```bash
    conda activate rl-examples
    ```

3. Go to the `rl_examples` directory:

    ```bash
    cd rl_examples
    ```

4. From the `rl_examples` directory, run the example, using the following command:

    ```bash
    python -m examples.<example_name>
    ```

    for example, to run the `party_or_relax` example, that is in the `dp` directory, run the following command from the `rl_examples` directory:

    ```bash
    python -m examples.dp.party_or_relax
    ```

## Available Examples

### Dynamic Programming

- [Party or Relax](rl_examples/examples/dp/party_or_relax.py) | [Problem description](https://artint.info/3e/html/ArtInt3e.Ch12.S5.html)
- [Stages Game](rl_examples/examples/dp/stages_game.py) | [Problem description](https://towardsdatascience.com/getting-started-with-markov-decision-processes-reinforcement-learning-ada7b4572ffb)
