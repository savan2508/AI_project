
# AI Project Repository: Reinforcement Learning Models for Game Environments

This repository includes implementations of three distinct AI models applying reinforcement learning to classic game environments:
- **Deep Q-Learning** for Pac-Man
- **Deep Q-Network (DQN)** for Lunar Lander
- **Asynchronous Advantage Actor-Critic (A3C)** for Kung Fu Master

Each project demonstrates an approach to solving control problems in games using deep reinforcement learning techniques.

## Table of Contents
- [Project Descriptions](#project-descriptions)
  - [Pac-Man Deep Q-Learning](#pac-man-deep-q-learning)
  - [Lunar Lander DQN](#lunar-lander-dqn)
  - [Kung Fu Master A3C](#kung-fu-master-a3c)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

## Project Descriptions

### Pac-Man Deep Q-Learning
**File**: `Pac_man_Deep_convolutional_Q_learning.ipynb`

This project implements a Deep Q-Learning (DQL) agent to play Pac-Man. The agent leverages convolutional neural networks to interpret the game environment and learn optimal moves by maximizing expected rewards.

**Key Features**:
- Environment setup and pre-processing
- Convolutional neural network for state-action evaluation
- Experience replay for stabilizing learning
- Q-value updates based on a discounted reward model

### Lunar Lander DQN
**File**: `Lunar_Landing.ipynb`

In this project, a Deep Q-Network (DQN) agent is trained to control the Lunar Lander. The goal is for the agent to safely land on a target platform, maximizing reward for smooth landing and penalizing crashes.

**Key Features**:
- Q-Network with target network to improve stability
- Reward shaping to encourage successful landing
- Experience replay to generalize learning
- Exploration-exploitation balancing with epsilon decay

### Kung Fu Master A3C
**File**: `kung_fu_A3C.ipynb`

This implementation applies the Asynchronous Advantage Actor-Critic (A3C) algorithm in the Kung Fu Master game environment. The A3C model runs multiple agents in parallel, optimizing both policy and value functions to enable complex decision-making.

**Key Features**:
- Parallel agent training for asynchronous updates
- Separate networks for policy (actor) and value (critic)
- Advantage function for stabilizing updates
- Multi-threaded environment setup

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/savan2508/AI_project.git
   cd AI_project
   ```

2. **Install Dependencies**:
   Install the necessary packages for reinforcement learning models:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To execute each project, open the corresponding Jupyter notebook (`.ipynb` file) and follow these steps:
1. **Initialize the Game Environment**: Set up the environment for training.
2. **Define and Configure the Neural Network**: Establish the architecture used for decision-making.
3. **Train the Agent**: Apply the chosen reinforcement learning algorithm to teach the agent.
4. **Evaluate Performance**: Assess how well the model performs within the game environment.

#### Example
To run the **Pac-Man Deep Q-Learning** project:
- Open `Pac_man_Deep_convolutional_Q_learning.ipynb`.
- Follow the cell-by-cell instructions to initialize, train, and evaluate the Pac-Man model.

### Results

Each notebook includes visualizations and logs that demonstrate the learning progress and performance of the agent. Key metrics like episode rewards, success rates, and Q-value trends provide insights into the model's effectiveness.

### Dependencies

The project requires the following main libraries:
- **TensorFlow/Keras**: For building and training deep learning models.
- **OpenAI Gym**: Provides reinforcement learning environments.
- **NumPy and Matplotlib**: For data manipulation and visualization.

To install these dependencies, run:
```bash
pip install -r requirements.txt
```

### License

This repository is licensed under the MIT License. Refer to the [LICENSE](LICENSE) file for additional details.

---

Each notebook contains step-by-step code explanations and training processes, offering a hands-on approach to experimenting with reinforcement learning.
