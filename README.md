# SARSA and Q-Learning for solving the cliff-walking problem

## Problem Statement

We have an agent trying to cross a 4 X 12 grid utilising on-policy (SARSA) and off-policy (Q-Learning) TD Control algorithms. The grid settings is as shown below:

![image](https://user-images.githubusercontent.com/22128902/92508405-39c34900-f226-11ea-83a8-eff4af5ffabb.png)

This is a standard un-discounted, episodic task, with start and goal states, and the usual actions causing movement up, down, right, and left. Reward is -1 on all transitions except those into the region marked Cliff. Stepping into this region incurs a reward of optimal path -100 and sends the agent instantly back to the start.

The state action value function Q(S,A) is updated by the agent according  to  SARSA or Q-Learning policy chosen. In the cliff walking example, the action set {UP, DOWN, LEFT, RIGHT} is chosen using exploration policies like Epsilon Greedy and Upper Confidence Bound.

## Epsilon Greedy Algorithm

The ε-greedy algorithm takes the best action most of the time, but does random exploration occasionally. The action value is estimated according to the past experience by averaging the rewards associated with the target action a that we have observed so far (up to the current time step t):

![image](https://user-images.githubusercontent.com/22128902/92508514-65deca00-f226-11ea-8dc8-fed491bf947a.png)

## UCB Algorithm:

In UCB algorithm, we always select the greediest action to maximize the upper confidence bound: 

A<sub>t</sub>= argmax<sub>a</sub>[ Q<sub>t</sub>(A) + c* sqrt( lnt/N<sub>t</sub>(A)) ]

## Q-Learning

Q-learning is an off-policy reinforcement learning algorithm that seeks to find the best action to take given the current state. It’s considered off-policy because the q-learning function learns from actions that are outside the current policy, like taking random actions, and therefore a policy isn’t needed. 

Q(S, A) <- Q(S,A) + α[R + 𝛾max<sub>a</sub>Q(S', a) - Q(S, A)]

Here the agent interacts with the environment and make updates to the state action pairs in our q-table Q[state, action].The agent interacts with the environment through exploration policy ε-greedy and UCB
The basic steps involved is: 
1.	Agent starts in a state (s1) takes an action (a1) and receives a reward (r1)
2.	Agent selects action by exploration policies
3.	Update q-values

## SARSA Algorithm

SARSA is an on-policy TD control method. A policy is a state-action pair tuple. An on-policy control method chooses the action for each state during learning by following a certain policy (mostly the one it is evaluating itself, like in policy iteration). We estimate Q<sub>π</sub>(s, a) for the current policy π and all state-action (s-a) pairs. We do this using TD update rule applied at every timestep by letting the agent transition from one state-action pair to another state-action pair

Q(S, A) <- Q(S,A) + α[R + 𝛾Q(S', A') - Q(S, A)]

The action A’ in the above algorithm is given by following e-greedy and UCB policy.
