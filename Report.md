# Tennis Agent
# Project's Goal

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

![alt text]()

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

In my implementation, I have chosen to solve the second version of the environment using the MADDPG algorithm. 
