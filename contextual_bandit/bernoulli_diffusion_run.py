from baselines.load_data import load_mnist_1d, load_yelp, load_movielen
from baselines.bernoulli_diffusion import BernoulliDiffusionBandit, PrioritizedReplayBuffer

import numpy as np
import copy
import os
import torch
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Bernoulli Diffusion Bandit')
    parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist', 'yelp', 'movielens'], help='Dataset choice')
    parser.add_argument('--beta', default=1.0, type=float, help='Exploration parameter')
    parser.add_argument('--n_samples', default=10, type=int, help='Samples per arm')
    parser.add_argument('--hidden_dim', default=128, type=int, help='Hidden layer dimension')
    parser.add_argument('--time_dim', default=32, type=int, help='Time encoding dimension')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--train_epochs', default=1, type=int, help='Epochs per training step')
    parser.add_argument('--n_timesteps', default=50, type=int, help='Diffusion time steps')
    parser.add_argument('--noise_schedule', default='linear', type=str, choices=['linear', 'cosine'], help='Noise schedule')
    parser.add_argument('--runs', default=3, type=int, help='Number of repeated experiments')
    parser.add_argument('--pretrain', default=500, type=int, help='Number of pretraining samples')
    parser.add_argument('--pretrain_epochs', default=25, type=int, help='Pretraining epochs')
    parser.add_argument('--visualize', default=True, type=bool, help='Whether to visualize results')
    parser.add_argument('--load_pretrained', action='store_true', help='Load pretrained model')
    parser.add_argument('--pretrained_path', default=None, type=str, help='Path to pretrained model')

    parser.add_argument('--use_buffer', action='store_true', help='Use prioritized replay buffer')
    parser.add_argument('--confidence_threshold', default=0.4, type=float, help='Confidence threshold')
    parser.add_argument('--hard_ratio', default=0.75, type=float, help='Hard sample ratio in replay buffer')

    args = parser.parse_args()

    if args.dataset == 'mnist':
        data_loader = load_mnist_1d()
    elif args.dataset == 'yelp':
        data_loader = load_yelp()
    elif args.dataset == 'movielens':
        data_loader = load_movielen()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Running Bernoulli Diffusion Bandit on dataset: {args.dataset}")
    print(f"Context dimension: {data_loader.dim}, Number of arms: {data_loader.n_arm}")

    regrets_all = []
    rewards_all = []

    for run in range(args.runs):
        print(f"Run {run+1}/{args.runs}")

        model = BernoulliDiffusionBandit(
            context_dim=data_loader.dim,
            n_timesteps=args.n_timesteps,
            noise_schedule=args.noise_schedule,
            hidden_dim=args.hidden_dim,
            time_dim=args.time_dim,
            lr=args.lr
        )
        replay_buffer = PrioritizedReplayBuffer(
            buffer_size=5000, 
            hard_sample_ratio=args.hard_ratio, 
            confidence_threshold=args.confidence_threshold
        )

        if args.load_pretrained and args.pretrained_path:
            print(f"Loading pretrained model from {args.pretrained_path}...")
            try:
                pretrained_state = torch.load(args.pretrained_path)
                model.model.load_state_dict(pretrained_state['model_state_dict'])
                model.optimizer.load_state_dict(pretrained_state['optimizer_state_dict'])
                if 'context_list' in pretrained_state and 'reward_list' in pretrained_state:
                    model.context_list = pretrained_state['context_list']
                    model.reward_list = pretrained_state['reward_list']
                print("Pretrained model loaded successfully")
                pretrained_model_state = copy.deepcopy(model.model.state_dict())
                pretrained_optimizer_state = copy.deepcopy(model.optimizer.state_dict())
                pretrained_context_list = copy.deepcopy(model.context_list)
                pretrained_reward_list = copy.deepcopy(model.reward_list)
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
                exit()    

        elif args.pretrain > 0:
            print(f"Starting pretraining with {args.pretrain} samples...")
            for i in range(args.pretrain):
                context, rwd = data_loader.step()
                arm = np.random.randint(0, data_loader.n_arm)
                reward = rwd[arm]
                model.update(context[arm], reward)
                if (i+1) % 100 == 0:
                    print(f"Pretraining sample collection: {i+1}/{args.pretrain}")
            
            print("Training with collected pretraining data...")
            for epoch in range(args.pretrain_epochs):
                loss = model.train(batch_size=min(args.batch_size, args.pretrain), epochs=5)
                print(f"Pretrain epoch {epoch+1}/{args.pretrain_epochs}, Loss: {loss:.4f}")
            
            checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f'{checkpoint_dir}/pretrained_model_{args.dataset}_epochs{args.pretrain_epochs}.pt'
            pretrained_state = {
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'context_list': copy.deepcopy(model.context_list),
                'reward_list': copy.deepcopy(model.reward_list)
            }
            torch.save(pretrained_state, checkpoint_path)
            print(f"Pretrained checkpoint saved to: {checkpoint_path}")

            pretrained_context_list = copy.deepcopy(model.context_list)
            pretrained_reward_list = copy.deepcopy(model.reward_list)
            pretrained_model_state = copy.deepcopy(model.model.state_dict())
            pretrained_optimizer_state = copy.deepcopy(model.optimizer.state_dict())

        regrets = []
        rewards = []
        sum_regret = 0
        sum_reward = 0
        print("Starting main experiment")
        print("Step; Cumulative Regret; Avg Regret per Step; Training Loss")

        for t in range(2000):
            context, rwd = data_loader.step()
            best_arm = np.argmax(rwd)

            arm_select, success_rates = model.select(context, beta=args.beta, n_samples=args.n_samples)
            print(f"Sample {t+1}: Context {context.shape}, Best arm {best_arm}, Selected arm {arm_select}, Success rate {success_rates[arm_select]:.4f}")

            reward = rwd[arm_select]
            regret = np.max(rwd) - reward
            sum_regret += regret
            sum_reward += reward
            regrets.append(sum_regret)
            rewards.append(sum_reward)
            binary_reward = 1.0 if reward > 0.5 else 0.0

            if args.use_buffer:
                replay_buffer.add(
                    context=context[arm_select], 
                    arm=arm_select, 
                    reward=binary_reward, 
                    confidence=success_rates[arm_select], 
                    best_arm=best_arm, 
                    all_rewards=rwd, 
                    all_contexts=context
                )

            model.update(context[arm_select], binary_reward)

            if t % 10 == 0:
                if not args.use_buffer and args.pretrain > 0:
                    model.model.load_state_dict(pretrained_model_state)
                    model.optimizer.load_state_dict(pretrained_optimizer_state)

                if args.use_buffer and len(replay_buffer) > 0:
                    model.context_list = []
                    model.reward_list = []
                    batch_size = min(args.batch_size, len(replay_buffer))
                    samples = replay_buffer.sample(batch_size)
                    
                    for sample in samples:
                        model.update(sample['context'], sample['reward'])

                        is_deceptive = not sample['is_best_arm'] and sample['confidence'] > 0.5

                        if is_deceptive:
                            print(f"Deceptive sample detected: confidence {sample['confidence']:.4f} but wrong choice, training more")
                            for _ in range(5):
                                model.update(sample['context'], sample['reward'])
                        elif sample['confidence'] < args.confidence_threshold:
                            for _ in range(2):
                                model.update(sample['context'], sample['reward'])

                loss = model.train(batch_size=min(args.batch_size, len(model.context_list)), epochs=args.train_epochs)
                print(f'{t}; {sum_regret}; {sum_regret/(t+1):.4f}; {loss:.4f}')
        
        print(f"Run {run+1} complete; Total Regret: {sum_regret}; Avg Regret: {sum_regret/10000:.4f}")
        regrets_all.append(regrets)
        rewards_all.append(rewards)

    regrets_all = np.array(regrets_all)
    rewards_all = np.array(rewards_all)

    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    if args.load_pretrained and args.pretrained_path:
        result_file = f'bernoulli_diffusion_{args.dataset}_beta{args.beta}_samples{args.n_samples}_pretrain{args.pretrained_path}_bufferimportant'
    else:
        result_file = f'bernoulli_diffusion_{args.dataset}_beta{args.beta}_samples{args.n_samples}_pretrain{args.pretrain}'
    np.save(f'{results_dir}/{result_file}_regrets.npy', regrets_all)
    np.save(f'{results_dir}/{result_file}_rewards.npy', rewards_all)
    print(f"Results saved to: {results_dir}/{result_file}_regrets.npy and {results_dir}/{result_file}_rewards.npy")

if args.visualize:
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    mean_regrets = np.mean(regrets_all, axis=0)
    std_regrets = np.std(regrets_all, axis=0)
    x = np.arange(len(mean_regrets))
    plt.plot(mean_regrets, label='Average Regret')
    plt.fill_between(x, mean_regrets - std_regrets, mean_regrets + std_regrets, alpha=0.3)
    plt.title(f'Cumulative Regret - {args.dataset}')
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Regret')
    plt.grid()

    plt.subplot(2, 2, 2)
    mean_rewards = np.mean(rewards_all, axis=0)
    std_rewards = np.std(rewards_all, axis=0)
    plt.plot(mean_rewards, label='Average Reward')
    plt.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
    plt.title(f'Cumulative Reward - {args.dataset}')
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Reward')
    plt.grid()

    plt.subplot(2, 2, 3)
    avg_regrets = mean_regrets / (np.arange(len(mean_regrets)) + 1)
    plt.plot(avg_regrets, label='Average Regret per Round')
    plt.title(f'Average Regret per Round - {args.dataset}')
    plt.xlabel('Rounds')
    plt.ylabel('Average Regret')
    plt.grid()

    plt.subplot(2, 2, 4)
    avg_rewards = mean_rewards / (np.arange(len(mean_rewards)) + 1)
    plt.plot(avg_rewards, label='Average Reward per Round')
    plt.title(f'Average Reward per Round - {args.dataset}')
    plt.xlabel('Rounds')
    plt.ylabel('Average Reward')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'{results_dir}/{result_file}_results.png')
    plt.show()
