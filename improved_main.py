import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import time
import argparse
import torch
import random
import json

# Import custom modules
from preprocessing import preprocess_data_for_rl, prepare_rl_environment_data
from improved_environment import WaterAllocationEnvironment
from improved_agent import WaterAllocationDQNAgent

def configure_logging():
    """
    Configure logging for the main experiment
    
    Returns:
    --------
    logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Set up logging to file and console
    log_file = f'logs/water_allocation_{time.strftime("%Y%m%d-%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_water_allocation_model(file_path, episodes=500, seed=42, model_save_path='models'):
    """
    Full pipeline for training water allocation RL model
    
    Parameters:
    -----------
    file_path : str
        Path to the input CSV file
    episodes : int, default=500
        Number of training episodes
    seed : int, default=42
        Random seed for reproducibility
    model_save_path : str, default='models'
        Directory to save trained models
    
    Returns:
    --------
    tuple: (trained agent, environment, training history)
    """
    # Configure logging
    logger = configure_logging()
    
    # Set random seeds for reproducibility
    set_seeds(seed)
    logger.info(f"Random seed set to {seed}")
    
    # Create model directory if it doesn't exist
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    try:
        start_time = time.time()
        logger.info(f"Starting water allocation training pipeline with {episodes} episodes")
        
        # Log hardware info
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("No GPU detected. Training will be slower on CPU.")
        
        # Step 1: Preprocess data
        logger.info("Preprocessing agricultural data...")
        rl_data = preprocess_data_for_rl(file_path)
        
        # Log preprocessing stats
        logger.info(f"Train set size: {rl_data['train'].shape}")
        logger.info(f"Validation set size: {rl_data['val'].shape}")
        logger.info(f"Test set size: {rl_data['test'].shape}")
        logger.info(f"Feature count: {len(rl_data['feature_names'])}")
        
        # Step 2: Prepare RL environment data
        logger.info("Preparing RL environment data...")
        env_data = prepare_rl_environment_data(rl_data)
        
        # Step 3: Create improved environment
        logger.info("Creating improved Water Allocation Environment...")
        env = WaterAllocationEnvironment(env_data)
        
        # Step 4: Create and train DQN agent
        logger.info("Initializing improved DQN Agent...")
        agent = WaterAllocationDQNAgent(
            state_dim=env.state_dim, 
            action_dim=len(env.action_space)
        )
        
        # Step 5: Train the agent
        logger.info(f"Starting agent training with {episodes} episodes...")
        training_history = agent.train(env, episodes=episodes)
        
        # Step 6: Evaluate the trained agent
        logger.info("Evaluating trained agent...")
        evaluation_metrics = agent.evaluate(env, episodes=100)
        
        # Step 7: Save evaluation metrics
        metrics_file = os.path.join(model_save_path, 'evaluation_metrics.json')
        with open(metrics_file, 'w') as f:
            # Convert numpy values to native Python types for JSON serialization
            serializable_metrics = {}
            for k, v in evaluation_metrics.items():
                if isinstance(v, dict):
                    serializable_metrics[k] = {str(k2): float(v2) for k2, v2 in v.items()}
                elif isinstance(v, (np.float32, np.float64)):
                    serializable_metrics[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    serializable_metrics[k] = int(v)
                else:
                    serializable_metrics[k] = v
            
            json.dump(serializable_metrics, f, indent=4)
        
        logger.info(f"Evaluation metrics saved to {metrics_file}")
        
        # Log total training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Visualize final results
        visualize_final_results(training_history, evaluation_metrics)
        
        return agent, env, training_history
    
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def visualize_final_results(training_history, evaluation_metrics):
    """
    Create comprehensive visualizations of training and evaluation results
    
    Parameters:
    -----------
    training_history : dict
        Training history from agent
    evaluation_metrics : dict
        Evaluation metrics from agent
    """
    # Create visualizations directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    logger = configure_logging()
    logger.info("Generating final visualizations...")
    
    # Plot reward comparison (training vs evaluation)
    plt.figure(figsize=(12, 8))
    
    # Training rewards (smoothed)
    if len(training_history['episode_rewards']) > 0:
        window_size = min(20, len(training_history['episode_rewards']) // 5)
        if len(training_history['episode_rewards']) > window_size:
            smoothed = np.convolve(
                training_history['episode_rewards'], 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            plt.plot(
                range(window_size-1, len(training_history['episode_rewards'])), 
                smoothed, 
                'b-', 
                linewidth=2,
                label='Training Rewards (Smoothed)'
            )
    
    # Add evaluation mean reward line
    if 'mean_reward' in evaluation_metrics:
        eval_mean = evaluation_metrics['mean_reward']
        eval_std = evaluation_metrics['std_reward']
        plt.axhline(y=eval_mean, color='r', linestyle='-', label=f'Evaluation Mean: {eval_mean:.2f}')
        plt.axhline(y=eval_mean + eval_std, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=eval_mean - eval_std, color='r', linestyle='--', alpha=0.5)
        plt.fill_between(
            range(len(training_history['episode_rewards'])),
            [eval_mean - eval_std] * len(training_history['episode_rewards']),
            [eval_mean + eval_std] * len(training_history['episode_rewards']),
            color='r', alpha=0.1
        )
    
    plt.title('Training vs Evaluation Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('visualizations/reward_comparison.png')
    plt.close()
    
    # Plot water allocation distribution (if available)
    if 'action_distribution' in evaluation_metrics:
        plt.figure(figsize=(10, 6))
        actions = list(evaluation_metrics['action_distribution'].keys())
        frequencies = list(evaluation_metrics['action_distribution'].values())
        
        # Sort by water amount
        sorted_data = sorted(zip(actions, frequencies))
        actions, frequencies = zip(*sorted_data)
        
        plt.bar(actions, frequencies, alpha=0.7)
        plt.title('Water Allocation Distribution During Evaluation')
        plt.xlabel('Water Amount')
        plt.ylabel('Frequency (%)')
        plt.grid(axis='y', alpha=0.3)
        
        # Add data labels
        for i, v in enumerate(frequencies):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        plt.savefig('visualizations/water_allocation_distribution.png')
        plt.close()
    
    logger.info("Final visualizations saved to 'visualizations/' directory")

def main():
    """
    Main function to run the water allocation RL experiment
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Water Allocation RL Experiment')
    parser.add_argument('--file', '-f', type=str, default="Crop_recommendationV2.csv",
                        help='Path to the input CSV file')
    parser.add_argument('--episodes', '-e', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--load', '-l', type=str, default=None,
                        help='Path to pretrained model to load (skip training)')
    parser.add_argument('--evaluate', '-ev', action='store_true',
                        help='Run evaluation only (requires --load)')
    args = parser.parse_args()
    
    # Configure logging
    logger = configure_logging()
    logger.info(f"Starting Water Allocation RL Experiment")
    logger.info(f"Arguments: {args}")
    
    # Set up directories
    for directory in ['models', 'logs', 'visualizations']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Check if evaluation only mode
    if args.evaluate:
        if args.load is None:
            logger.error("Evaluation mode requires a model path (--load)")
            return
        
        # Load model and evaluate
        logger.info(f"Loading model from {args.load} for evaluation...")
        
        # Set random seeds
        set_seeds(args.seed)
        
        try:
            # Preprocess data
            logger.info("Preprocessing agricultural data...")
            rl_data = preprocess_data_for_rl(args.file)
            
            # Prepare environment data
            logger.info("Preparing RL environment data...")
            env_data = prepare_rl_environment_data(rl_data)
            
            # Create environment
            logger.info("Creating evaluation environment...")
            env = WaterAllocationEnvironment(env_data)
            
            # Create agent and load model
            agent = WaterAllocationDQNAgent(
                state_dim=env.state_dim, 
                action_dim=len(env.action_space)
            )
            agent.load(args.load)
            
            # Evaluate
            logger.info("Evaluating model...")
            evaluation_metrics = agent.evaluate(env, episodes=100)
            
            # Save evaluation metrics
            metrics_file = 'evaluation_metrics.json'
            with open(metrics_file, 'w') as f:
                # Convert numpy values to native Python types for JSON serialization
                serializable_metrics = {}
                for k, v in evaluation_metrics.items():
                    if isinstance(v, dict):
                        serializable_metrics[k] = {str(k2): float(v2) for k2, v2 in v.items()}
                    elif isinstance(v, (np.float32, np.float64)):
                        serializable_metrics[k] = float(v)
                    elif isinstance(v, (np.int32, np.int64)):
                        serializable_metrics[k] = int(v)
                    else:
                        serializable_metrics[k] = v
                
                json.dump(serializable_metrics, f, indent=4)
            
            logger.info(f"Evaluation metrics saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # File path for the dataset
    file_path = args.file
    
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found at: {file_path}")
        logger.info("Please provide the correct path to your dataset with --file argument.")
        return
    
    # Check if loading pretrained model
    if args.load is not None:
        logger.info(f"Loading pretrained model from {args.load}")
        
        # Will be implemented later if needed
        # For now, proceed with normal training
        
    # Start training process
    logger.info(f"Starting water allocation RL experiment with {args.episodes} episodes")
    start_time = time.time()
    
    agent, env, training_history = train_water_allocation_model(
        file_path=file_path,
        episodes=args.episodes,
        seed=args.seed
    )
    
    # Log total experiment time
    total_time = time.time() - start_time
    logger.info(f"Water Allocation RL Experiment completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Water Allocation RL Experiment Complete! Total time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()