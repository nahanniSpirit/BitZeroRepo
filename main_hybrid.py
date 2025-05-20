"""
BitZero Main Interface with Hybrid Precision

This module provides the main interface for the BitZero proof-of-concept,
integrating the hybrid precision components with task generation and learning systems.
"""

import torch
import numpy as np
import argparse
import os
import time
import json
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

from hybrid_precision import HybridPrecisionTransformer, create_bitzero_nano_hybrid, create_bitzero_micro_hybrid
from task_generator import TaskGenerator, TaskVerifier


class HybridReinforcementLearningTrainer:
    """Reinforcement learning trainer for BitZero with hybrid precision."""
    
    def __init__(self, 
                 model: HybridPrecisionTransformer,
                 task_generator: TaskGenerator,
                 task_verifier: TaskVerifier,
                 learning_rate: float = 1e-5,
                 max_seq_len: int = 768,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize reinforcement learning trainer.
        
        Args:
            model: BitZero transformer model with hybrid precision
            task_generator: Task generator
            task_verifier: Task verifier
            learning_rate: Learning rate for optimizer
            max_seq_len: Maximum sequence length
            device: Device to use for training
        """
        self.model = model
        self.task_generator = task_generator
        self.task_verifier = task_verifier
        self.device = device
        self.max_seq_len = max_seq_len
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training statistics
        self.stats = {
            "episodes": 0,
            "correct_answers": 0,
            "total_answers": 0,
            "rewards": [],
            "difficulties": []
        }
        
        # Create checkpoint directory
        os.makedirs("checkpoints", exist_ok=True)
    
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Simple tokenization for proof-of-concept.
        
        Args:
            text: Input text
            
        Returns:
            Tensor of token IDs
        """
        # This is a simplified tokenization for the proof-of-concept
        # In a real implementation, would use a proper tokenizer
        tokens = []
        for char in text:
            # Map character to a token ID (simple ASCII mapping)
            token_id = ord(char) % 256
            tokens.append(token_id)
        
        # Pad or truncate to max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def detokenize(self, token_ids: torch.Tensor) -> str:
        """
        Simple detokenization for proof-of-concept.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Decoded text
        """
        # Convert token IDs back to characters
        text = ""
        for token_id in token_ids.squeeze().tolist():
            if token_id > 0:  # Skip padding tokens
                text += chr(token_id % 256)
        
        return text
    
    def generate_answer(self, task: str, max_new_tokens: int = 100) -> str:
        """
        Generate an answer for a given task.
        
        Args:
            task: Task description
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated answer
        """
        # Tokenize task
        input_ids = self.tokenize(task)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate answer token by token
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.model(input_ids)
                
                # Get next token prediction (from last position)
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature and top-k sampling
                temperature = 0.7
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k sampling
                top_k = 40
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                
                # Sample from the probability distribution
                next_token_id = top_k_indices[torch.multinomial(probs, num_samples=1)]
                
                # Ensure next_token_id has the right shape [1, 1]
                next_token_id = next_token_id.view(1, 1)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
                
                # Check if we've reached the end of the sequence
                if next_token_id.item() == 0:  # End token
                    break
        
        # Detokenize to get the answer
        # Extract only the newly generated tokens (skip the original task)
        original_len = len(task)
        generated_tokens = input_ids[0, original_len:]
        answer = self.detokenize(generated_tokens)
        
        return answer
    
    def train_step(self, task: str, answer: str, reward: float) -> Dict[str, float]:
        """
        Perform a single training step using reinforcement learning.
        
        Args:
            task: Task description
            answer: Generated answer
            reward: Reward signal
            
        Returns:
            Dictionary with training metrics
        """
        # Tokenize task and answer
        input_ids = self.tokenize(task)
        target_ids = self.tokenize(answer)
        
        # Set model to training mode
        self.model.train()
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Compute loss with reward scaling
        # This is a simplified approach; in practice would use a proper RL algorithm like PPO
        loss = 0
        for i in range(min(len(answer), self.max_seq_len)):
            if i < logits.size(1):
                token_logits = logits[0, i, :]
                target_id = target_ids[0, i]
                token_loss = torch.nn.functional.cross_entropy(token_logits.unsqueeze(0), target_id.unsqueeze(0))
                # Scale loss by reward (higher reward = lower loss)
                loss += token_loss * (1.0 - reward)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "reward": reward
        }
    
    def train_episode(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Train for a single episode (generate task, answer, verify, learn).
        
        Args:
            task_type: Optional task type ("math" or "code")
            
        Returns:
            Dictionary with episode metrics
        """
        # Generate task
        task, expected_answer, task_type = self.task_generator.generate_task(task_type)
        
        # Generate answer
        answer = self.generate_answer(task)
        
        # Verify answer
        is_correct, confidence = self.task_verifier.verify(
            task, answer, expected_answer, task_type
        )
        
        # Calculate reward
        reward = float(is_correct) * confidence
        
        # Train on this example
        metrics = self.train_step(task, answer, reward)
        
        # Update statistics
        self.stats["episodes"] += 1
        self.stats["total_answers"] += 1
        if is_correct:
            self.stats["correct_answers"] += 1
        self.stats["rewards"].append(reward)
        self.stats["difficulties"].append(self.task_generator.current_difficulty)
        
        # Adjust difficulty based on recent performance
        if self.stats["episodes"] % 10 == 0:
            recent_success_rate = sum(self.stats["rewards"][-10:]) / 10
            self.task_generator.adjust_difficulty(recent_success_rate)
        
        # Return episode information
        return {
            "task": task,
            "answer": answer,
            "expected_answer": expected_answer,
            "task_type": task_type,
            "is_correct": is_correct,
            "confidence": confidence,
            "reward": reward,
            "difficulty": self.task_generator.current_difficulty,
            "loss": metrics["loss"]
        }
    
    def train(self, 
              num_episodes: int, 
              eval_interval: int = 10,
              save_interval: int = 100,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Train the model for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to train for
            eval_interval: Interval for evaluation and reporting
            save_interval: Interval for saving checkpoints
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training statistics
        """
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            # Alternate between math and code tasks
            task_type = "math" if episode % 2 == 1 else "code"
            
            # Train for one episode
            episode_info = self.train_episode(task_type)
            
            # Print progress
            if verbose and episode % eval_interval == 0:
                elapsed_time = time.time() - start_time
                success_rate = self.stats["correct_answers"] / max(1, self.stats["total_answers"])
                avg_reward = sum(self.stats["rewards"][-eval_interval:]) / eval_interval
                
                print(f"Episode {episode}/{num_episodes} | "
                      f"Time: {elapsed_time:.2f}s | "
                      f"Success Rate: {success_rate:.4f} | "
                      f"Avg Reward: {avg_reward:.4f} | "
                      f"Difficulty: {self.task_generator.current_difficulty:.2f}")
                
                print(f"Task: {episode_info['task'][:100]}...")
                print(f"Answer: {episode_info['answer'][:100]}...")
                print(f"Correct: {episode_info['is_correct']}")
                print("-" * 50)
            
            # Save checkpoint
            if episode % save_interval == 0:
                self.save_checkpoint(f"checkpoints/bitzero_hybrid_episode_{episode}.pt")
        
        # Final statistics
        final_stats = {
            "episodes": self.stats["episodes"],
            "success_rate": self.stats["correct_answers"] / max(1, self.stats["total_answers"]),
            "final_difficulty": self.task_generator.current_difficulty,
            "training_time": time.time() - start_time,
            "quantization_stats": self.model.get_quantization_stats()
        }
        
        return final_stats
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": self.stats,
            "task_generator_difficulty": self.task_generator.current_difficulty
        }
        
        torch.save(checkpoint, path)
        
        # Also save stats as JSON
        stats_path = path.replace(".pt", "_stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "episodes": self.stats["episodes"],
                "success_rate": self.stats["correct_answers"] / max(1, self.stats["total_answers"]),
                "difficulties": self.stats["difficulties"][-100:],  # Last 100 difficulties
                "rewards": self.stats["rewards"][-100:],  # Last 100 rewards
                "quantization_stats": self.model.get_quantization_stats()
            }, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.stats = checkpoint["stats"]
        self.task_generator.current_difficulty = checkpoint["task_generator_difficulty"]


class HybridPersonalizationManager:
    """Manager for BitZero personalization with hybrid precision."""
    
    def __init__(self, 
                 model_path: str,
                 save_dir: str = "personalization",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_seq_len: int = 768): # Added max_seq_len
        """
        Initialize personalization manager.
        
        Args:
            model_path: Path to base model checkpoint
            save_dir: Directory to save personalization data
            device: Device to use
            max_seq_len: Maximum sequence length for the personalization model
        """
        self.model_path = model_path
        self.save_dir = save_dir
        self.device = device
        self.max_seq_len = max_seq_len # Store max_seq_len
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load base model
        # Pass max_seq_len to model creation
        self.model = create_bitzero_nano_hybrid(critical_ratio=0.1, max_seq_len=self.max_seq_len).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Initialize task generator and verifier
        self.task_generator = TaskGenerator()
        self.task_verifier = TaskVerifier()
        
        # Initialize trainer
        self.trainer = HybridReinforcementLearningTrainer(
            model=self.model,
            task_generator=self.task_generator,
            task_verifier=self.task_verifier,
            device=device,
            max_seq_len=self.max_seq_len # Pass max_seq_len to trainer
        )
    
    def add_conversation(self, user_input: str, model_response: str):
        """
        Add conversation to history.
        
        Args:
            user_input: User input
            model_response: Model response
        """
        self.conversation_history.append({
            "user": user_input,
            "model": model_response,
            "timestamp": time.time()
        })
    
    def personalize(self, num_episodes: int = 10, verbose: bool = True) -> Dict[str, Any]:
        """
        Personalize the model based on conversation history.
        
        Args:
            num_episodes: Number of episodes to train for
            verbose: Whether to print progress
            
        Returns:
            Dictionary with personalization statistics
        """
        if not self.conversation_history:
            return {"status": "No conversation history to personalize from"}
        
        # Extract tasks from conversation history
        tasks = []
        for conv in self.conversation_history[-10:]:  # Use last 10 conversations
            user_input = conv["user"]
            model_response = conv["model"]
            
            # Create a task from the conversation
            task = f"User: {user_input}\nAssistant:"
            answer = model_response
            
            tasks.append((task, answer))
        
        # Train on these tasks
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            # Select a random task from history
            task, answer = random.choice(tasks)
            
            # Tokenize
            input_ids = self.trainer.tokenize(task)
            
            # Set model to training mode
            self.model.train()
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Compute loss (simple language modeling loss)
            target_ids = self.trainer.tokenize(answer)
            loss = 0
            for i in range(min(len(answer), self.trainer.max_seq_len)):
                if i < logits.size(1):
                    token_logits = logits[0, i, :]
                    target_id = target_ids[0, i]
                    token_loss = torch.nn.functional.cross_entropy(token_logits.unsqueeze(0), target_id.unsqueeze(0))
                    loss += token_loss
            
            # Backward pass and optimization
            self.trainer.optimizer.zero_grad()
            loss.backward()
            self.trainer.optimizer.step()
            
            if verbose and episode % 5 == 0:
                print(f"Personalization Episode {episode}/{num_episodes} | Loss: {loss.item():.4f}")
        
        # Save personalized model
        timestamp = int(time.time())
        save_path = f"{self.save_dir}/personalized_model_{timestamp}.pt"
        self.save(save_path)
        
        return {
            "status": "Personalization complete",
            "episodes": num_episodes,
            "training_time": time.time() - start_time,
            "save_path": save_path
        }
    
    def save(self, path: str):
        """
        Save personalized model.
        
        Args:
            path: Path to save model
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "conversation_history": self.conversation_history,
            "timestamp": time.time()
        }
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """
        Load personalized model.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.conversation_history = checkpoint.get("conversation_history", [])


class BitZeroHybridSystem:
    """Main system for BitZero with hybrid precision."""
    
    def __init__(self, 
                 model_size: str = "nano", 
                 critical_ratio: float = 0.1,
                 max_seq_len: int = 768,  # Added max_seq_len
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize BitZero hybrid system.
        
        Args:
            model_size: Size of the model ("nano" or "micro")
            critical_ratio: Critical ratio for hybrid precision
            max_seq_len: Maximum sequence length for the model
            device: Device to use
        """
        self.model_size = model_size
        self.critical_ratio = critical_ratio
        self.device = device
        self.max_seq_len = max_seq_len # Store max_seq_len
        
        # Create model
        if model_size.lower() == "nano":
            self.model = create_bitzero_nano_hybrid(critical_ratio=critical_ratio, max_seq_len=self.max_seq_len)
        elif model_size.lower() == "micro":
            self.model = create_bitzero_micro_hybrid(critical_ratio=critical_ratio, max_seq_len=self.max_seq_len)
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        self.model.to(self.device)
        
        # Initialize components
        self.task_generator = TaskGenerator(initial_difficulty=0.2)
        self.task_verifier = TaskVerifier()
        
        self.trainer = HybridReinforcementLearningTrainer(
            model=self.model,
            task_generator=self.task_generator,
            task_verifier=self.task_verifier,
            device=device
        )
        
        self.personalization_manager = None  # Initialize later when needed
        
        print(f"BitZero Hybrid system initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"Quantization stats: {self.model.get_quantization_stats()}")
    
    def train(self, 
              num_episodes: int = 100, 
              eval_interval: int = 10,
              save_interval: int = 50,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            num_episodes: Number of episodes to train for
            eval_interval: Interval for evaluation and reporting
            save_interval: Interval for saving checkpoints
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training statistics
        """
        print(f"Starting training for {num_episodes} episodes...")
        start_time = time.time()
        
        # Train the model
        stats = self.trainer.train(
            num_episodes=num_episodes,
            eval_interval=eval_interval,
            save_interval=save_interval,
            verbose=verbose
        )
        
        # Save final checkpoint
        final_path = f"{self.checkpoint_dir}/bitzero_hybrid_{self.model_size}_final.pt"
        self.trainer.save_checkpoint(final_path)
        
        # Plot training progress
        self._plot_training_progress()
        
        print(f"Training completed in {time.time() - start_time:.2f}s")
        print(f"Final checkpoint saved to {final_path}")
        
        return stats
    
    def _plot_training_progress(self):
        """Plot training progress and save figures."""
        # Extract data
        episodes = list(range(1, len(self.trainer.stats["rewards"]) + 1))
        rewards = self.trainer.stats["rewards"]
        difficulties = self.trainer.stats["difficulties"]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot rewards
        ax1.plot(episodes, rewards, 'b-')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.grid(True)
        
        # Plot difficulties
        ax2.plot(episodes, difficulties, 'r-')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Difficulty')
        ax2.set_title('Task Difficulty')
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/training_progress_hybrid.png")
        plt.close()
    
    def evaluate(self, num_tasks: int = 20, task_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a set of tasks.
        
        Args:
            num_tasks: Number of tasks to evaluate on
            task_types: Optional list of task types to evaluate on
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating model on {num_tasks} tasks...")
        
        if task_types is None:
            task_types = ["math", "code"]
        
        results = {
            "total_tasks": num_tasks,
            "correct": 0,
            "task_types": {t: {"total": 0, "correct": 0} for t in task_types}
        }
        
        # Set model to evaluation mode
        self.model.eval()
        
        for i in range(num_tasks):
            # Alternate between task types
            task_type = task_types[i % len(task_types)]
            
            # Generate task
            task, expected_answer, task_type = self.task_generator.generate_task(task_type)
            
            # Generate answer
            answer = self.trainer.generate_answer(task)
            
            # Verify answer
            is_correct, confidence = self.task_verifier.verify(
                task, answer, expected_answer, task_type
            )
            
            # Update results
            results["task_types"][task_type]["total"] += 1
            if is_correct:
                results["correct"] += 1
                results["task_types"][task_type]["correct"] += 1
            
            if i % 5 == 0 or i == num_tasks - 1:
                print(f"Progress: {i+1}/{num_tasks} tasks evaluated")
        
        # Calculate success rates
        results["success_rate"] = results["correct"] / results["total_tasks"]
        for task_type in task_types:
            type_total = results["task_types"][task_type]["total"]
            type_correct = results["task_types"][task_type]["correct"]
            if type_total > 0:
                results["task_types"][task_type]["success_rate"] = type_correct / type_total
            else:
                results["task_types"][task_type]["success_rate"] = 0.0
        
        # Print results
        print(f"Evaluation complete: {results['success_rate']:.4f} success rate")
        for task_type in task_types:
            print(f"  {task_type}: {results['task_types'][task_type]['success_rate']:.4f} "
                  f"({results['task_types'][task_type]['correct']}/{results['task_types'][task_type]['total']})")
        
        # Save results
        with open(f"{self.log_dir}/evaluation_results_hybrid.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def personalize(self, 
                    conversation_history: List[Dict[str, str]],
                    num_episodes: int = 20,
                    verbose: bool = True) -> Dict[str, Any]:
        """
        Personalize the model based on conversation history.
        
        Args:
            conversation_history: List of conversation dictionaries with "user" and "model" keys
            num_episodes: Number of episodes to train for
            verbose: Whether to print progress
            
        Returns:
            Dictionary with personalization statistics
        """
        print("Initializing personalization...")
        
        # Initialize personalization manager if not already done
        if self.personalization_manager is None:
            # Use the latest checkpoint if available
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)))
                checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
            else:
                # No checkpoint available, use the current model state
                checkpoint_path = f"{self.checkpoint_dir}/temp_checkpoint_hybrid.pt"
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                    "stats": self.trainer.stats,
                    "task_generator_difficulty": self.task_generator.current_difficulty
                }, checkpoint_path)
            
            self.personalization_manager = HybridPersonalizationManager(
                model_path=checkpoint_path,
                save_dir=f"{self.checkpoint_dir}/personalized_hybrid",
                device=self.device,
                max_seq_len=self.max_seq_len # Pass max_seq_len
            )
        
        # Add conversations to history
        for conv in conversation_history:
            self.personalization_manager.add_conversation(conv["user"], conv["model"])
        
        # Personalize the model
        stats = self.personalization_manager.personalize(
            num_episodes=num_episodes,
            verbose=verbose
        )
        
        print(f"Personalization complete: {stats['status']}")
        if "save_path" in stats:
            print(f"Personalized model saved to {stats['save_path']}")
        
        return stats
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.trainer.load_checkpoint(checkpoint_path)
        print("Checkpoint loaded successfully")
    
    def save_checkpoint(self, checkpoint_path: str):
        """
        Save model to checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint to
        """
        print(f"Saving checkpoint to {checkpoint_path}...")
        self.trainer.save_checkpoint(checkpoint_path)
        print("Checkpoint saved successfully")


def main():
    """Main function for BitZero Hybrid CLI."""
    parser = argparse.ArgumentParser(description="BitZero Hybrid: Quantized Self-Play Micro Model")
    parser.add_argument("--model_size", type=str, default="nano", choices=["nano", "micro"], help="Model size")
    parser.add_argument("--critical_ratio", type=float, default=0.1, help="Critical ratio for hybrid precision")
    parser.add_argument("--max_seq_len", type=int, default=768, help="Maximum sequence length for the model") # Added max_seq_len argument
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--action", type=str, required=True, 
                        choices=["train", "evaluate", "personalize"],
                        help="Action to perform")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes for training")
    parser.add_argument("--tasks", type=int, default=20,
                        help="Number of tasks for evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize BitZero system
    system = BitZeroHybridSystem(
        model_size=args.model_size, 
        critical_ratio=args.critical_ratio,
        max_seq_len=args.max_seq_len, # Pass max_seq_len
        device=args.device
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        system.load_checkpoint(args.checkpoint)
    
    # Perform action
    if args.action == "train":
        system.train(
            num_episodes=args.episodes,
            verbose=args.verbose
        )
    elif args.action == "evaluate":
        system.evaluate(
            num_tasks=args.tasks
        )
    elif args.action == "personalize":
        # For CLI, use a simple example conversation
        example_conversation = [
            {"user": "What is the capital of France?", "model": "The capital of France is Paris."},
            {"user": "How do I calculate the area of a circle?", "model": "To calculate the area of a circle, use the formula A = πr², where r is the radius of the circle."}
        ]
        
        system.personalize(
            conversation_history=example_conversation,
            num_episodes=args.episodes,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()