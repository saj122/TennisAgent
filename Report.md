# Tennis Agent
# Project's Goal

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

![alt text](https://github.com/saj122/TennisAgent/blob/master/images/tennis.gif)

# Environment Details

The environment is based on Unity ML-agents. The project environment provided by Udacity is similar to the Tennis environment on the Unity ML-Agents GitHub page.

    The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and 
    simulations to serve as environments for training intelligent agents. Agents can be trained using reinforcement 
    learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

In my implementation, I have chosen to solve the environment using the MADDPG algorithm. 

# Algorithm

The algorithm used in solving the environment is the Deep Deterministic Policy Gradient (DDPG). DDPG is an algorithm which learns a Q-function and a policy.

![alt text](https://github.com/saj122/ReacherAgent/blob/master/images/ddpg.png)

The algorithm image was taken from a Towards Data Science [article](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287).

# Code Implementation

The code used here is derived from the DDPG bipedal tutorial from the Deep Reinforcement Learning Nanodegree.

The code is written in Python 3.6 and is relying on PyTorch 0.4.0 framework.

The code implements an Actor and Critic classes.

The Actor and Critic classes each implement a Target and a Local Neural Networks used for the training.

Instead of an Agent class, I created two actors. One for each agent. Both actors shared the same replay buffer and critic network. Both actors act on the state and contribute to the same replay buffer. They sample from that buffer when learning.

Gradient clipping of the critics parameters hindered the progress of learning. The agents learned faster without it.

The algorithm and environment was very sensitive to noise. A low sigma and seed helped in learning.
   
# DDPG Hyperparameters

BUFFER_SIZE = int(1e6)  # replay buffer size

BATCH_SIZE = 128        # minibatch size

GAMMA = 0.99            # discount factor

TAU = 2e-1              # for soft update of target parameters

LR_ACTOR = 1e-4         # learning rate of the actor 

LR_CRITIC = 3e-4        # learning rate of the critic

WEIGHT_DECAY = 0        # L2 weight decay

SEED = 2  

#### Actor Neural Network
Input nodes (24) -> Fully Connected Layer (512 nodes, Relu activation) -> Fully Connected Layer (256 nodes, Relu activation) -> Fully Connected Layer (256 nodes, Relu activation) -> Ouput nodes (2, tanh activation)

#### Critic Neural Network
Input nodes (24) -> Fully Connected Layer (512 nodes, Relu activation) -> Fully Connected Layer (256 + 4 nodes, Relu activation) -> Fully Connected Layer (256 nodes, Relu activation) - > Fully Connected Layer (256 nodes, Relu activation) - >Ouput nodes (1, relu activation)

# Results
Given the hyperparameters and neural network the agent was able to achieve an average reward of 0.5 in 400 episodes.

![alt text](https://github.com/saj122/TennisAgent/blob/master/images/scores.png)

# Future Work

Using other algorithms, such as, PPO along with the actor and critic paradigm.

Researching new advancements as multi agent deep reinforcment learning is fairly new and exploding field.

Checking out deepminds new OpenAI Five and the recently research Starcraft 2 AI.
