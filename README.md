# Reinforcement learning for Control

## Stochastic Policy Search

### Policy network
In the same way as we used data-driven optimization to tune the gains $K_P,K_I,K_D$ in the PID controllers, we can use the same approach to tune (or train) the parameters (or weights) $\boldsymbol{\theta}$ of a neural network.

In the first part a relatively simple neural network controller in PyTorch is built.

### Reinforce algorithm 

In order to implement and apply the Reinforce algorithm, the following steps are performed:

*   Create a [policy network](#policy_net) that uses transfer learning
*   Create an auxiliary function that selects [control actions](#control_actions) out of the distribution
*   Create an auxilary function that runs [multiple episodes](#multi_episodes) per epoch
*   Finally, put all the pieces together into a function that computes the [Reinforce algorithm](#r_alg)

