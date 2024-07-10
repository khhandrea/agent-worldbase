# Lightning-RL
- Reinforcement learning framework with lightning trainer interface
- Want to be integrated with Trainer someday

## TODO
- save wandb at same file or split

## Directories
- `lightning-sample`: lightning tutorial
- `lightning-rl`: lightning-RL implementaion

## `lightning-sample`
```python
python3 main.py
```

## `lightning-RL`
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
ppo_agent = PPOAgent(actor_critic_network)

env = gym.make("CartPole-v0")

trainer = AgentTrainer()
trainer.fit(ppo_agent, env)
```

### Train world model
```python

random_dataset = make_random_trajectory(env, directory=directory)
random_trajectory = DataLoader(random_trajectory)
trainer = Trainer()
trainer.fit(agent, dataset=random_dataloader)
```

### Abstracting
```python
import lightning as L

if laerning_type.startswith("RL"):
    trainer = AgentTrainer()
    target = env
else:
    trainer = Trainer()
    target = L.Dataloader(mnist_dataset)

model = MLP()

trainer.fit(model, train_dataloader)
```

## Implementation (pseudocode)
### Agent class (inheriting LightningAgentModule)
```python
class Agent(L.RL.LightningAgentModule):
    ...
    def bind_env(self, env):
        self._env = env

    def initial_rollout(self):
        obs, info = self._env.reset()
        rollout(total_step)

    def rollout(self, total_step):
        if self.step == 0:
            obs, info = self._env.reset()
        for _ in total_step:
            action = self.network.policy(obs)
            obs, reward, terminated, truncated, info = self._env.step(action)
            self.replay_buffer.add(obs, action, reward, terminated)

    def training_step(self, batch):
        obs, action, reward, terminated = batch
        ...
        loss = actor_loss + critic_loss + entropy_loss
        return loss

    def sample(self):
        return self._replay_buffer.sample()
```

### AgentTrainer
```python
class AgentTrainer(Trainer):
    ...

    def fit(self, agent: LightningModule, train_dataloader: EnvDataloader):
        while self.step < self.max_step
            agent.rollout()
            batch = agent.replay_buffer.sample()
            loss = agent.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```