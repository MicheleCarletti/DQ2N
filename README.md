# DQ2N
This repository is related to Parma University Master Thesis in Computer Science Engineering: "A Quantum Reinforcement Learning Framework for Optimal Control Problem Solving".

It provides a Quantum Reinforcement Learning framework called Deep Quantum Q-Network (DQ2N), to solve optimal control problems, like the OpenAI Gymnasium CartPole environment. The model is tained via Q-learning algorithm.

`vqc_cartpole.py` contains the DQ2N implementation, the training and other ancillary functions.

`Prioritized_experience_replay.py` implements the PER buffer, used by the Q-learning algorithm.

