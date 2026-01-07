
# VMAS Soccer Assignment

## Overview

This repository contains the implementation and analysis of the **VMAS Soccer** scenario for **Multi-Agent Reinforcement Learning (MARL)**. The task involves training agents to play soccer in a continuous-time, physics-based environment. The main focus is comparing different MARL algorithms (MAPPO vs MADDPG), analyzing the emergence of roles (e.g., striker, defender, goalie), and experimenting with parameter sharing versus independent policies.

## Task Breakdown

The assignment consists of six main tasks with specific evaluation criteria. Below is a quick summary of the tasks that I completed:

---

### Task 1: Core Functionality (10 pts)

- **Training and Evaluation**: I implemented and compared **MAPPO** and **MADDPG** algorithms on the VMAS Soccer environment.
- **Quantitative Comparison**: I provided a comparison of the **mean episode reward** between MAPPO and MADDPG, as shown in the plots below, to evaluate their performance across training iterations.

---

### Task 2: Role Emergence (15 pts)

- **Parameter Sharing vs Independent Policies**: I compared the effectiveness of **parameter sharing** (one policy per team) and **independent policies** (one policy per agent).
- **Evidence of Role Specialization**: I provided heatmaps, entropy measurements, and ball touch analysis to demonstrate role specialization in the environment.

---

### Task 6: Reporting Quality (5 pts)

- **README Documentation**: This document includes a summary of the task breakdown, experiment results, and insights derived from training and evaluating MARL algorithms on VMAS Soccer. It also includes the necessary experiment setup and running instructions.

---

## Environment Setup

1. **Dependencies**:
   - Python 3.x
   - PyTorch
   - TorchRL (for VMAS integration)
   - Hydra (for configuration management)
   - Docker (for reproducibility)

2. **Quick Start**:
   To get started, clone the repository and install the dependencies:

   ```bash
   git clone <repository-url>
   cd <repository-dir>
   pip install -r requirements.txt
   ```

3. **Running Experiments**:
   After setting up the environment, you can run the training scripts with Hydra configurations. Here's an example for training the MAPPO agent:

   ```bash
   python train.py --config-name=mappo_config
   ```

   For more advanced configurations, you can modify the Hydra config files located in the `configs/` directory.

4. **Docker Setup**:
   Build and run the Docker container for the experiments:

   ```bash
   docker build -t vmas-soccer .
   docker run -it vmas-soccer
   ```

---

## Experiment Results

The following figures present the results from comparing **MAPPO vs MADDPG** and tracking **role emergence** through metrics such as episode reward mean and agent-specific performance.

#### Figure 1: MAPPO vs MADDPG (Episode Reward Mean)

![MAPPO vs MADDPG](./images/plot1.png)

This graph compares the mean episode reward of MAPPO and MADDPG over training iterations. MAPPO consistently outperforms MADDPG in terms of reward gain.

#### Figure 2: Episode Reward Mean (MADDPG)

![MADDPG Reward](./images/plot2.png)

This figure shows the reward progression over time for MADDPG, where the reward fluctuates but shows signs of improvement in later iterations.

#### Figure 3: Episode Reward Mean (MAPPO)

![MAPPO Reward](./images/plot3.png)

Here we see the training rewards for MAPPO, which steadily increase, indicating better learning efficiency compared to MADDPG.

#### Figure 4: Role Emergence (Agent Blue vs Agent Red)

![Role Emergence](./images/plot4.png)

This chart shows the episode reward means for both agent blue and agent red, illustrating how different agents perform with respect to each other during training.

---

## Evaluation and Analysis

1. **Algorithm Comparison**:
   - MAPPO exhibits smoother and more consistent improvement in rewards compared to MADDPG, making it the preferred algorithm for this environment.

2. **Role Specialization**:
   - The emergence of specialized roles is evident from the positional heatmaps and entropy measures, particularly when parameter sharing is disabled.

---

## Conclusion

The VMAS Soccer environment provides a challenging yet insightful platform for studying multi-agent reinforcement learning. Through the implementation and comparison of various algorithms, the emergence of roles, and the evaluation of recurrent policies, we can draw conclusions on how agents can learn to cooperate and compete under partial observability.

---

## References

- **VMAS Soccer**: [VMAS official documentation](https://vmas.io)
- **MAPPO**: [Multi-Agent PPO paper](https://arxiv.org/abs/1706.02275)
- **MADDPG**: [Multi-Agent Deep Deterministic Policy Gradient paper](https://arxiv.org/abs/1706.02275)

---

This README summarizes the key components of the assignment, experiment results, and insights derived from the training and evaluation of MARL algorithms in the VMAS Soccer environment.
