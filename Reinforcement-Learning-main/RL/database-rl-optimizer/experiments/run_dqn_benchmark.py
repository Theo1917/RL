"""End-to-end DQN benchmark comparing tabular Q-learning, DQN, and PostgreSQL."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from src.database import generate_database
from src.query import generate_query
from src.environment import JoinOrderEnv
from src.q_learning import QLearningAgent
from src.deep_rl.dqn_agent import DQNAgent, DoubleDQNAgent
from src.baselines import exhaustive_optimal


def train_tabular_agent(env, episodes=500, seed=42):
    """Train tabular Q-learning agent."""
    agent = QLearningAgent(
        num_tables=len(env.query.tables),
        mode='per_query',
        epsilon=1.0,
        epsilon_decay=0.995,
        alpha=0.1,
        gamma=0.99,
        seed=seed
    )
    
    rewards = []
    for episode in tqdm(range(episodes), desc="Training Tabular Q"):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            valid_actions = info['valid_actions']
            action = agent.choose_action(env, obs, valid_actions)
            next_obs, reward, done, _, info = env.step(action)
            
            agent.update(obs, action, reward, next_obs, info['valid_actions'], done)
            
            obs = next_obs
            episode_reward += reward
        
        agent.decay_epsilon()
        rewards.append(episode_reward)
    
    return agent, rewards


def train_dqn_agent(env, episodes=500, agent_type='dqn', seed=42):
    """Train DQN or Double DQN agent."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if agent_type == 'double_dqn':
        agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            network_type='standard',
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            batch_size=32,
            buffer_capacity=10000,
            target_update_freq=50,
            seed=seed
        )
    else:
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            network_type='standard',
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            batch_size=32,
            buffer_capacity=10000,
            target_update_freq=50,
            seed=seed
        )
    
    rewards = []
    losses = []
    
    for episode in tqdm(range(episodes), desc=f"Training {agent_type.upper()}"):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            valid_actions = info['valid_actions']
            action = agent.select_action(obs, valid_actions, training=True)
            next_obs, reward, done, _, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done, info['valid_actions'])
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            
            obs = next_obs
            episode_reward += reward
        
        agent.update_epsilon()
        rewards.append(episode_reward)
    
    return agent, rewards, losses


def evaluate_agent(agent, env, num_episodes=10, agent_type='tabular'):
    """Evaluate trained agent."""
    costs = []
    optimal_costs = []
    
    for _ in range(num_episodes):
        # Agent's plan
        obs, info = env.reset()
        done = False
        
        while not done:
            valid_actions = info['valid_actions']
            
            if agent_type == 'tabular':
                action = agent.choose_action(env, obs, valid_actions, training=False)
            else:  # DQN
                action = agent.select_action(obs, valid_actions, training=False)
            
            obs, reward, done, _, info = env.step(action)
        
        agent_cost = info['estimated_cost']
        costs.append(agent_cost)
        
        # Optimal plan
        optimal_plan, optimal_cost = exhaustive_optimal(env)
        optimal_costs.append(optimal_cost)
    
    costs = np.array(costs)
    optimal_costs = np.array(optimal_costs)
    ratios = costs / optimal_costs
    
    return {
        'mean_cost': costs.mean(),
        'mean_optimal': optimal_costs.mean(),
        'mean_ratio': ratios.mean(),
        'median_ratio': np.median(ratios),
        'std_ratio': ratios.std()
    }


def run_comparison_experiment(num_tables=4, episodes=500, eval_episodes=20, seed=42):
    """Run comprehensive comparison experiment."""
    print(f"\n{'='*60}")
    print(f"Running DQN vs Tabular Q-Learning Comparison")
    print(f"Tables: {num_tables}, Training Episodes: {episodes}")
    print(f"{'='*60}\n")
    
    # Generate database and query
    np.random.seed(seed)
    database = generate_database(num_tables=num_tables, seed=seed)
    query = generate_query(database, seed=seed)
    
    # Create environment
    env = JoinOrderEnv(query, database, start_mode='fixed', reward_mode='cost')
    
    results = {
        'config': {
            'num_tables': num_tables,
            'episodes': episodes,
            'eval_episodes': eval_episodes,
            'seed': seed
        },
        'agents': {}
    }
    
    # Train and evaluate each agent
    agents_to_test = [
        ('tabular', train_tabular_agent),
        ('dqn', lambda env, ep, s: train_dqn_agent(env, ep, 'dqn', s)),
        ('double_dqn', lambda env, ep, s: train_dqn_agent(env, ep, 'double_dqn', s))
    ]
    
    for agent_name, train_fn in agents_to_test:
        print(f"\n--- Training {agent_name.upper()} ---")
        
        if agent_name == 'tabular':
            agent, rewards = train_fn(env, episodes, seed)
            losses = []
        else:
            agent, rewards, losses = train_fn(env, episodes, seed)
        
        print(f"✓ Training complete")
        
        # Evaluate
        print(f"Evaluating on {eval_episodes} test episodes...")
        eval_results = evaluate_agent(agent, env, eval_episodes, 
                                     'tabular' if agent_name == 'tabular' else 'dqn')
        
        results['agents'][agent_name] = {
            'training_rewards': rewards,
            'losses': losses,
            'evaluation': eval_results
        }
        
        print(f"✓ Mean cost ratio vs optimal: {eval_results['mean_ratio']:.4f}")
    
    return results


def plot_results(results, output_dir='results/plots'):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Learning curves
    ax = axes[0, 0]
    for agent_name, data in results['agents'].items():
        rewards = data['training_rewards']
        # Smooth with moving average
        window = 50
        smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean()
        ax.plot(smoothed, label=agent_name.upper(), linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('Learning Curves (Smoothed)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Loss curves (DQN only)
    ax = axes[0, 1]
    for agent_name, data in results['agents'].items():
        if data['losses']:
            losses = data['losses']
            window = 100
            smoothed = pd.Series(losses).rolling(window, min_periods=1).mean()
            ax.plot(smoothed, label=agent_name.upper(), linewidth=2)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss (DQN Agents)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 3. Final performance comparison
    ax = axes[1, 0]
    agent_names = list(results['agents'].keys())
    ratios = [results['agents'][name]['evaluation']['mean_ratio'] for name in agent_names]
    stds = [results['agents'][name]['evaluation']['std_ratio'] for name in agent_names]
    
    bars = ax.bar(agent_names, ratios, yerr=stds, capsize=5, alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Optimal', linewidth=2)
    ax.set_ylabel('Cost Ratio vs Optimal', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Color bars
    colors = ['skyblue', 'orange', 'green']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    for name in agent_names:
        eval_data = results['agents'][name]['evaluation']
        table_data.append([
            name.upper(),
            f"{eval_data['mean_ratio']:.4f}",
            f"{eval_data['median_ratio']:.4f}",
            f"{eval_data['std_ratio']:.4f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Agent', 'Mean Ratio', 'Median Ratio', 'Std Dev'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0.3, 1, 0.5]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_file = output_dir / 'dqn_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plots to {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='DQN vs Tabular Q-Learning Benchmark')
    parser.add_argument('--tables', type=int, default=5, help='Number of tables')
    parser.add_argument('--episodes', type=int, default=500, help='Training episodes')
    parser.add_argument('--eval-episodes', type=int, default=20, help='Evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/dqn_benchmark.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Run experiment
    start_time = time.time()
    results = run_comparison_experiment(
        num_tables=args.tables,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed
    )
    
    results['runtime_seconds'] = time.time() - start_time
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\n✓ Saved results to {output_path}")
    
    # Generate plots
    plot_results(results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Tables: {results['config']['num_tables']}")
    print(f"  Training Episodes: {results['config']['episodes']}")
    print(f"  Runtime: {results['runtime_seconds']:.1f}s")
    print(f"\nPerformance (Cost Ratio vs Optimal):")
    
    for agent_name in ['tabular', 'dqn', 'double_dqn']:
        if agent_name in results['agents']:
            eval_data = results['agents'][agent_name]['evaluation']
            print(f"  {agent_name.upper():12s}: {eval_data['mean_ratio']:.4f} ± {eval_data['std_ratio']:.4f}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
