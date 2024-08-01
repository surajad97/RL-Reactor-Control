### Reinforcement learning for Control

## Stochastic Policy Search

# Policy network
In the same way as we used data-driven optimization to tune the gains $K_P,K_I,K_D$ in the PID controllers, we can use the same approach to tune (or train) the parameters (or weights) $\boldsymbol{\theta}$ of a neural network.

In the first part a relatively simple neural network controller in PyTorch is built.

$${\bf u}:=\pi({\bf x};\boldsymbol{\theta})$$

and hard code a simple stochastic search algorithm (it is a combination of random search and local random search) to manipulate the weights $\boldsymbol{\theta}$, evaluate the performance of the current weight values, and iterate.

**Neural Network Controller Training Algorithm**

*Initialization*

Collect $d$ initial datapoints $\mathcal{D}=\{(\hat{f}^{(j)}=\sum_{k=0}^{k=T_f} (e(k))^2,~ \boldsymbol{\theta}^{(j)}) \}_{j=0}^{j=d}$ by simulating $x(k+1) = f(x(\cdot),u(\cdot))$ for different values of $\boldsymbol{\theta}^{(j)}$, set a small radious of search $r$

*Main loop*

1. *Repeat*
2. $~~~~~~$ Choose best current known parameter value $\boldsymbol{\theta}^*$.
3. $~~~~~~$ Sample $n_s$ values around $\boldsymbol{\theta}^*$, that are at most some distance $r$, $\bar{\boldsymbol{\theta}}^{(0)},...,\bar{\boldsymbol{\theta}}^{(n_s)}$
3. $~~~~~~$ Simulate new values  $ x(k+1) = f(x(k),u(\bar{\boldsymbol{\theta}}^{(i)};x(k))), ~ k=0,...,T_f-1, i=0,...,n_s $
4. $~~~~~~$ Compute $\hat{f}^{(i)}=\sum_{k=0}^{k=T_f} (e(k))^2, i=0,...,n_s$.
5. $~~~~~~$ **if** $\bar{\boldsymbol{\theta}}^{\text{best}}$ is better than $\boldsymbol{\theta}^*$, then $ \boldsymbol{\theta}^* \leftarrow \bar{\boldsymbol{\theta}}^{\text{best}}$, **else** $ r \leftarrow r\gamma$, where $ 0 < \gamma <1 $ 
6. until stopping criterion is met.
