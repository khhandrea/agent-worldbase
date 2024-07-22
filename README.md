# Agent-WorldBase
- Reinforcement learning framework with lightning trainer interface
- Want to be integrated with Trainer

## TODO
- save wandb at same file or split

## Directories
- `lightning-sample`: lightning tutorial
- `lightning-rl`: lightning-RL implementaion

## Example
```python
python3 main.py
```

## Pseudocode
### Default
```python
from lightning import Trainer
from lightning.RL import AgentTrainer, TrajectoryDataset
from lightning.RL import make_random_trajectory
from torch.utils.data import DataLoader
```

### Train online-RL agent
```python
actor_critic_network = ActorCriticNetwork()
replay_buffer = ReplayBuffer()
ppo_agent_config = PPOAgentConfig(
    actor_critic_network=actor_critic_network,
    replay_buffer=replay_buffer,
)
ppo_agent = PPOAgent(ppo_agent_config)

env = gym.make("CartPole-v0")

agent_trainer = AgentTrainer()
agent_trainer.fit(ppo_agent, env)
```

### AgentTrainer
```python
class AgentTrainer(Trainer):
    ...

    def fit(self, agent: LightningModule, train_dataloader: EnvDataloader):
        while self.step < self.max_step
            for _ in range(rollout_num):
                action = self.agent.get_action(obs)
                obs, reward, truncated, terminated, info = env.step(action)
                agent.batch(obs, action, reward)
            
            batch = agent.sample_batch()
            loss = agent.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.step += 1
```