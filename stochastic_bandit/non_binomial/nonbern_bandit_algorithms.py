import numpy as np
import tqdm
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

import warnings
warnings.filterwarnings("ignore")


def get_reward(theta):
    p_0, p_half = theta
    # with p_0 to get 0, with p_half to get 0.5, otherwise get 1
    return np.random.choice([0, 0.5, 1], p=[p_0, p_half, 1 - p_0 - p_half])


class NonBern_Bandit_Algorithm:
    def __init__(self, N, T, theta_oracled, trajectory_path=None):
        self.N = N
        self.T = T
        self.theta_oracled = theta_oracled
        self.bandit_num = len(theta_oracled)
        self.reward_expect = np.zeros(self.bandit_num)
        for i in range(self.bandit_num):
            self.reward_expect[i] = 1 - theta_oracled[i][0] - theta_oracled[i][1] / 2

        self.best_arm = np.argmax(self.reward_expect)

        self.regret = None
        self.reward = None
        self.optimal_choice = None
        self.name = None
        self.label_name = None
        self.param_list = None
        self.algorithm = None

        self.trajectory = None
        if trajectory_path is not None:
            self.trajectory = np.load(trajectory_path)
        self.checkflag = False


    def __call__(self):
        print(f"Running {self.name} Algorithm...")

        for i, param in enumerate(self.param_list):

            rewards_sum  = np.zeros(self.N)
            optimal_sum  = np.zeros(self.N)

            for _ in tqdm.tqdm(range(self.T)):
                reward, optimal_choice = self.algorithm(param)   # shape (N,)
                rewards_sum  += reward
                optimal_sum  += optimal_choice

            self.reward[i, :]         = rewards_sum / self.T
            self.optimal_choice[i, :] = optimal_sum / self.T

            step_regret = np.maximum(0, self.reward_expect[self.best_arm] - self.reward[i, :])
            self.regret[i, :] = np.cumsum(step_regret)

            self.optimal_choice[i, :] = (np.cumsum(self.optimal_choice[i, :]) / np.arange(1, self.N + 1))



class UCB_Algorithm(NonBern_Bandit_Algorithm):
    def __init__(self, N, T, theta_oracled, c_list, trajectory_path=None):
        super().__init__(N, T, theta_oracled, trajectory_path)
        self.param_list = c_list
        self.optimal_choice = np.zeros((len(c_list), N), dtype=np.float64)
        self.reward = np.zeros((len(c_list), N))
        self.regret = np.zeros((len(c_list), N))
        self.name='UCB'
        self.label_name='c'
        self.algorithm = self.UCB

        self.count = np.zeros(self.bandit_num)
        self.theta = np.zeros(self.bandit_num)
        self.count, self.theta = self.train_offline_data(self.count, self.theta)

    def train_offline_data(self, count, theta):
        if self.trajectory is None:
            return count, theta
        traj_num, act_num, _ = self.trajectory.shape
        for i in range(traj_num):
            for t in range(act_num):

                selected_arm, reward = int(self.trajectory[i, t, 0]), self.trajectory[i, t, 1]

                count[selected_arm] += 1
                theta[selected_arm] += 1 / count[selected_arm] * (reward - theta[selected_arm])

        if self.checkflag == False:
            self.checkflag = True
            print('training offline data')
        return count, theta


    def UCB(self, c):
        count = self.count.copy()
        theta = self.theta.copy()

        reward = np.zeros(self.N)
        optimal_choice = np.zeros(self.N, dtype=np.float64)

        for t in range(self.N):
            # initialization
            if t < self.bandit_num:
                selected_arm = t
            else:
                selected_arm = np.argmax(theta + c * np.sqrt(2 * np.log(t) / count))

            r_i = get_reward(self.theta_oracled[selected_arm])
            count[selected_arm] += 1
            theta[selected_arm] += 1 / count[selected_arm] * (r_i - theta[selected_arm])

            reward[t] = self.reward_expect[selected_arm]
            if selected_arm == self.best_arm:
                optimal_choice[t] = np.float64(1)

        return reward, optimal_choice


class Gradient_Bandit_Algorithm(NonBern_Bandit_Algorithm):
    def __init__(self, N, T, theta_oracled, b_beta_list, trajectory_path=None):
        super().__init__(N, T, theta_oracled, trajectory_path)
        self.param_list = b_beta_list
        self.optimal_choice = np.zeros((len(b_beta_list), N), dtype=np.float64)
        self.reward = np.zeros((len(b_beta_list), N))
        self.regret = np.zeros((len(b_beta_list), N))
        self.name='Gradient Bandit'
        self.label_name='$b,\\beta$'
        self.algorithm = self.gradient

        self.H = np.zeros(self.bandit_num)
        self.H = self.train_offline_data(self.H, b_beta_list[0][0], b_beta_list[0][1])

    def softmax(self, H, beta, t):
        if type(beta) == str:
            beta_0, beta_T = 1, 10
            # beta = beta_0 + np.log(1 + 9 * t / self.N) / np.log(10) * (beta_T - beta_0)
            beta =  1.0 + 9.0 * t / max(self.N - 1, 1)  # starts at 1 ends near 10
        maxn = np.max(H)
        exp_H = np.exp(beta * (H - maxn))
        return exp_H / np.sum(exp_H)


    def get_action(self, policy, random_num):
        selected_arm = 0
        # this could be optimized by using binary search, but bandit_num = 3, it is not worth it
        for i in range(self.bandit_num):
            if random_num <= policy[i]:
                selected_arm = i
                break
            random_num -= policy[i]
        return selected_arm

    def train_offline_data(self, H, b, beta):
        if self.trajectory is None:
            return H
        # print(self.trajectory.shape)
        traj_num, act_num, _ = self.trajectory.shape
        alpha = 0.1
        average_reward = 0
        t = 0
        for i in range(traj_num):
            # if i > 2000 and i % 2000 == 0:
            #     alpha = alpha * 0.1
            # if np.max(self.H) > 20:
            #     self.H = self.H / 20
            # if np.max(self.H) > 50:
            #     break

            for iter in range(act_num):
                selected_arm, reward = int(self.trajectory[i, iter, 0]), self.trajectory[i, iter, 1]
                policy = self.softmax(H, beta, iter)

                # update H
                average_reward = t / (t + 1) * average_reward + 1 / (t + 1) * reward # incremental implementation
                t += 1
                if b != -1: # baseline
                    average_reward = b
                # if np.max(self.H) > 50:
                #     continue
                H += alpha * (reward - average_reward) * (np.eye(self.bandit_num)[selected_arm] - policy)

                # if H has nan
                if np.isnan(H).any():
                    print(f'{i = }, {iter = }, {H = }')

        if self.checkflag == False:
            self.checkflag = True
            print(f'{policy = }')
            print('training offline data')
        return H

    def gradient(self, b_beta):
        reward = np.zeros(self.N)
        optimal_choice = np.zeros(self.N, dtype=np.float64)

        random_num = np.random.uniform(0, 1, self.N)
        b, beta = b_beta

        H = self.H.copy()

        alpha = 0.1
        average_reward = 0

        for t in range(self.N):
            if self.trajectory is not None:
                policy = self.softmax(H, 'linear', t)
            else:
                policy = self.softmax(H, beta, t)
            selected_arm = self.get_action(policy, random_num[t])

            r_i = get_reward(self.theta_oracled[selected_arm])
            reward[t] = self.reward_expect[selected_arm]
            if selected_arm == self.best_arm:
                optimal_choice[t] = np.float64(1)

            # update H
            average_reward = t / (t + 1) * average_reward + 1 / (t + 1) * r_i # incremental implementation
            if b != -1: # baseline
                average_reward = b
            H += alpha * (r_i - average_reward) * (np.eye(self.bandit_num)[selected_arm] - policy)

        return reward, optimal_choice


class Offline_Policy_Gradient_Algorithm(NonBern_Bandit_Algorithm):
    def __init__(self, N, T, theta_oracled, b_beta_list, trajectory_path=None):
        super().__init__(N, T, theta_oracled, trajectory_path)
        self.param_list = b_beta_list
        self.optimal_choice = np.zeros((len(b_beta_list), N), dtype=np.float64)
        self.reward = np.zeros((len(b_beta_list), N))
        self.regret = np.zeros((len(b_beta_list), N))
        self.name='Offline Policy Gradient'
        self.label_name='$b,\\beta$'
        self.algorithm = self.policy_gradient

        self.EPOCHS: int = 2000
        self.LR: float = 0.1
        self.BETA_SCHEDULE: str = "linear"
        self.VERBOSE: int = 400

        self.H = np.zeros(self.bandit_num)
        self.train_offline_data(self.trajectory, epochs=self.EPOCHS, verbose=self.VERBOSE)

    # beta schedule
    def beta_t(self, t: int, T: int) -> float:
        if self.trajectory is not None and self.trajectory.shape[0] < 150:
            return 1.5
        # if self.beta_schedule == "linear":
        return 0.3
        # return 1.0 + 9.0 * t / max(T - 1, 1)  # starts at 1 ends near 10
        # return 0.5
    def softmax(self, H, beta, t):
        if type(beta) == str:
            beta_0, beta_T = 1, 10
            # beta = beta_0 + np.log(1 + 9 * t / self.N) / np.log(10) * (beta_T - beta_0)
            # beta =  5.0 + 5.0 * t / max(self.T - 1, 1)  # starts at 5 ends near 10
            beta = 10.0 + 10.0 * t / max(self.N - 1, 1)

        maxn = np.max(H)
        exp_H = np.exp(beta * (H - maxn))
        return exp_H / np.sum(exp_H)

    def get_action(self, policy, random_num):
        selected_arm = 0
        # this could be optimized by using binary search, but bandit_num = 3, it is not worth it
        for i in range(self.bandit_num):
            if random_num <= policy[i]:
                selected_arm = i
                break
            random_num -= policy[i]
        return selected_arm

    def finetune(self, traj_arr: np.ndarray, *, epochs: int = 1000, verbose: int = 100) -> None:
        # finetune with the undiffused data
        print('finetune')
        m, T, _ = traj_arr.shape
        if m == 100:
            return
        m = 100
        actions = traj_arr[:100, :, 0].astype(int)
        rewards = traj_arr[:100, :, 1]
        # lr = 1
        for ep in tqdm.tqdm(range(3000)):
            lr = 1 / np.sqrt(ep + 1)
            # if np.max(self.H) > 20:
            #     self.H = self.H / 20
            # if np.max(self.H) > 50:
            #     break
            for i in range(m):
                acts  = actions[i]
                rews  = rewards[i]
                G = rews[::-1].cumsum()[::-1]    # cumulative future reward
                average_reward = 0
                batch_size = 8
                grad = 0
                for t in range(T):
                    beta_t = self.beta_t(t, T)
                    pi_t = self.softmax(self.H, beta_t, t)
                    # grad += (G[t] - average_reward) * beta_t * (np.eye(self.bandit_num)[acts[t]] - pi_t)
                    grad += (rews[t] - average_reward) * beta_t * (np.eye(self.bandit_num)[acts[t]] - pi_t)
                    if t % batch_size == 0 or t == T - 1:
                        self.H += lr * grad
                        grad = 0
                    # if np.max(self.H) > 50:
                    #     break
                    # self.H += lr * (G[t] - average_reward) * beta_t * (np.eye(self.bandit_num)[acts[t]] - pi_t)
                    # self.H += lr * (rews[t] - average_reward) * beta_t * (np.eye(self.bandit_num)[acts[t]] - pi_t)
                    average_reward = t / (t + 1) * average_reward + 1 / (t + 1) * rews[t]
            self.H -= np.max(self.H)


    # main training loop
    def train_offline_data(self, traj_arr: np.ndarray, *, epochs: int = 1000, verbose: int = 100) -> None:
        m, T, _ = traj_arr.shape
        actions = traj_arr[:, :, 0].astype(int)
        rewards = traj_arr[:, :, 1]
        # print(f'{rewards = }')
        # lr = self.LR
        batch_size = 32
        for ep in tqdm.tqdm(range(1, epochs + 1)):
            lr = 0.1

            old_theta = self.H.copy()
            for i in range(m):
                acts  = actions[i]
                rews  = rewards[i]
                G = rews[::-1].cumsum()[::-1]    # cumulative future reward
                average_reward = 0
                grad = 0
                for t in range(T):
                    beta_t = self.beta_t(t, T)
                    pi_t = self.softmax(self.H, beta_t, t)

                    grad += (rews[t] - average_reward) * beta_t * (np.eye(self.bandit_num)[acts[t]] - pi_t)
                    if t % batch_size == 0 or t == T - 1:
                        self.H += lr * grad
                        grad = 0

                    average_reward = t / (t + 1) * average_reward + 1 / (t + 1) * rews[t]

            if np.isnan(self.H).any():
                print(f'{ep = }, {self.H = }')

            # check convergence
            if np.allclose(old_theta, self.H, atol=1e-5):
                print(f"Converged after {ep} epochs.")
                break
        self.H -= np.max(self.H)

        print('Policy Gradient trained policy before finetuning: ', self.H)


    def policy_gradient(self, b_beta):
        reward = np.zeros(self.N)
        optimal_choice = np.zeros(self.N, dtype=np.float64)

        random_num = np.random.uniform(0, 1, self.N)
        b, beta = b_beta

        H = self.H.copy()

        # alpha = 5
        average_reward = 0

        for t in range(self.N):
            alpha = 1 / np.sqrt(t + 1)

            policy = self.softmax(H, beta, t)
            selected_arm = self.get_action(policy, random_num[t])

            r_i = get_reward(self.theta_oracled[selected_arm])
            reward[t] = self.reward_expect[selected_arm]
            if selected_arm == self.best_arm:
                optimal_choice[t] = np.float64(1)

            # update H
            average_reward = t / (t + 1) * average_reward + 1 / (t + 1) * r_i # incremental implementation
            if b != -1: # baseline
                average_reward = b
            H += alpha * (r_i - average_reward) * (np.eye(self.bandit_num)[selected_arm] - policy)

        return reward, optimal_choice


def report_regrets(algorithms):
    regrets = []
    for algorithm in algorithms:
        regrets.append(algorithm.regret)

    # plot regrets
    plt.figure(figsize=(10, 6))
    for i, algorithm in enumerate(algorithms):
        plt.plot(algorithm.regret.mean(axis=0), label=algorithm.name)
    # plt.xlabel("Turns")
    # plt.ylabel("Average Regret")
    # plt.title("Average Regret of Different Algorithms")
    plt.xlabel("轮次")
    plt.ylabel("平均累计遗憾值")
    plt.title("不同算法间的平均累计遗憾值比较")
    plt.legend()
    plt.grid(True)
    plt.gcf().set_dpi(300)
    plt.show()


def overall_analysis(algorithms):
    regrets = []
    for algorithm, name in algorithms:
        regrets.append(algorithm.regret)

    # plot regrets
    plt.figure(figsize=(10, 6))
    for algorithm, name in algorithms:
        plt.plot(algorithm.regret.mean(axis=0), label=name)
    # plt.xlabel("Turns")
    # plt.ylabel("Average Regret")
    # plt.title("Average Regret of Different Algorithms")
    plt.xlabel("轮次")
    plt.ylabel("平均累计遗憾值")
    plt.title("不同算法间的平均累计遗憾值比较")
    plt.legend()
    plt.grid(True)
    plt.gcf().set_dpi(300)
    plt.show()

    for i, (algorithm, name) in enumerate(algorithms):
        print(f"{name:30} : {algorithm.regret.mean(axis=0)[-1]:.3f}")