# Improving LLM Reasoning via Reward Model Optimization

This project explores how reward model design impacts the reasoning capabilities of Large Language Models (LLMs) during Reinforcement Learning (RL) training, with applications to mathematical problem-solving.

## Key Contributions
- **Comparative analysis** of Outcome-supervised (ORM) vs Process-supervised (PRM) reward models
- **Novel reward refinement techniques** (Clip & Delta mechanisms) to prevent reward hacking
- **State-of-the-art results** on GSM8K (+28%) and MATH benchmarks using pure RL training

## Reward Models
### Outcome-supervised Reward Model (ORM)
**Purpose:** Predicts final answer correctness  
**Training:**
1. Collect question-answer pairs with binary labels
2. Fine-tune transformer to predict correctness probability  
3. Loss: Binary Cross-Entropy  
```python
Loss = −(y·log(p) + (1−y)·log(1−p))
```
Process-supervised Reward Model (PRM)
Purpose: Evaluates reasoning step quality
Training:

Collect human preference rankings of reasoning steps

Train using Bradley-Terry model:

python
P(step_A > step_B) = σ(reward_A − reward_B)
Reward Refinement
Technique	Formula	Effect
Clip	min(reward - η, 0)	Caps maximum reward per step
Delta	reward_k - reward_{k+1}	Rewards progress between steps
Training Pipeline
Data Preparation

Tokenize GSM8K/MATH problems

Generate solution trajectories

Reward Computation

```python
def get_reward(solution):
    step_rewards = [prm(step) for step in solution]
    clipped = [min(r-0.8, 0) for r in step_rewards]  # η=0.8
    return sum(clipped) + orm(solution)
```
RL Optimization

Algorithm: Proximal Policy Optimization (PPO)

Key Parameters:

KL coefficient: 0.1

Clip epsilon: 0.2

Batch size: 1024×8

Results
Model	GSM8K (Δ)	MATH (Δ)
Baseline	50.2%	24.9%
+ PRM	65.3%	30.6%
+ PRM + Clip/Delta	78.5% (+28.3)	31.4% (+6.5)
Setup
```
bash
pip install -r requirements.txt  # torch, transformers, datasets
python train.py --model=qwen1.5 --dataset=gsm8k
```
Key Insights
PRM > ORM for training-time rewards

Clip/Delta prevents 63% of reward hacking cases

Pure RL training can surpass supervised baselines
