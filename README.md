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
