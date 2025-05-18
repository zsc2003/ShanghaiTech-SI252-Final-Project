# ShanghaiTech-SI252-Final-Project
ShanghaiTech SI252 Reinforcement Learning final project, Spring 2025.

# Weekly Report

## week 1 (2025/5/18)
- Contextual bandit: We've finished the basic code for fitting the reward distribution using discrete diffusion. The initial results look okay, but online training hasn’t shown much improvement yet, so we still need to tweak the training strategy.

- Stochastic bandit: We've finished the basic code for generating the trajectory: $\psi=((a_1,r_1),\dots,(a_T,r_T))$. And the corresponding code with different discrete diffusion methods to enlarge the offline dataset. Current method for generating the trajectory is using the Bernoulli reward, and Thomson sampling. And the generation methods are suing generating by pair, generating by sequence, and generating by sequence with transformer encoder. More generation methods and non-Bernoulli reward are under discovering.