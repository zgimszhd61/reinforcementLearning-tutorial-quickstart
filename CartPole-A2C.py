import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置环境
env = gym.make('CartPole-v1')

# 超参数
gamma = 0.99
lr = 0.001

# Actor-Critic 网络定义
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        policy = torch.softmax(self.policy(x), dim=-1)
        value = self.value(x)
        return policy, value

# 初始化网络、优化器
input_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = ActorCritic(input_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练函数
def train():
    state = env.reset()
    log_probs = []
    values = []
    rewards = []

    done = False
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, value = model(state)
        action = torch.multinomial(policy, 1).item()
        log_prob = torch.log(policy.squeeze(0)[action])
        
        next_state, reward, done, _ = env.step(action)
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        
        state = next_state

    return log_probs, values, rewards

# 更新函数
def update(log_probs, values, rewards):
    rewards = np.array(rewards)
    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
    R = sum(rewards * discounts[:-1])
    
    policy_loss = []
    value_loss = []
    returns = []
    G = 0
    for r in rewards[::-1]:
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    for log_prob, value, R in zip(log_probs, values, returns):
        advantage = R - value.item()
        policy_loss.append(-log_prob * advantage)
        value_loss.append(nn.functional.mse_loss(value, torch.tensor([R])))
        
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

# 主训练循环
num_episodes = 1000
for episode in range(num_episodes):
    log_probs, values, rewards = train()
    update(log_probs, values, rewards)
    if episode % 100 == 0:
        print(f'Episode {episode}, total reward: {sum(rewards)}')

env.close()
