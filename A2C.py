# 导入必要的库
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),  # 输入层到隐藏层的线性变换
            nn.ReLU(),  # 隐藏层的激活函数
            nn.Linear(128, action_size),  # 隐藏层到输出层的线性变换
            nn.Softmax(dim=-1)  # 输出层的激活函数，用于生成动作的概率分布
        )
    
    def forward(self, state):
        return self.network(state)  # 前向传播，根据状态生成动作的概率分布

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),  # 输入层到隐藏层的线性变换
            nn.ReLU(),  # 隐藏层的激活函数
            nn.Linear(128, 1)  # 隐藏层到输出层的线性变换，输出一个值作为状态价值
        )
    
    def forward(self, state):
        return self.network(state)  # 前向传播，根据状态评估价值

def train(env, actor, critic, actor_optimizer, critic_optimizer, episodes):
    for episode in range(episodes):
        state = env.reset()  # 重置环境，开始新的一轮
        done = False
        total_reward = 0
        
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)  # 将状态转换为张量
            action_probs = actor(state)  # 生成动作的概率分布
            distribution = torch.distributions.Categorical(action_probs)  # 根据概率分布创建一个分布对象
            action = distribution.sample()  # 从分布中采样一个动作
            
            next_state, reward, done, _ = env.step(action.item())  # 执行动作，获取下一个状态和奖励
            total_reward += reward
            
            # Critic评估
            state_value = critic(state)  # 当前状态的价值
            next_state_value = critic(torch.FloatTensor(next_state).unsqueeze(0))  # 下一个状态的价值
            target_value = reward + (0.99 * next_state_value)  # 目标价值，考虑折扣因子
            advantage = target_value - state_value  # 优势函数
            
            # 更新Critic
            critic_loss = advantage.pow(2).mean()  # 计算Critic的损失
            critic_optimizer.zero_grad()  # 清空梯度
            critic_loss.backward()  # 反向传播
            critic_optimizer.step()  # 更新Critic的参数
            
            # 更新Actor
            actor_loss = -(distribution.log_prob(action) * advantage.detach())  # 计算Actor的损失
            actor_optimizer.zero_grad()  # 清空梯度
            actor_loss.backward()  # 反向传播
            actor_optimizer.step()  # 更新Actor的参数
            
            state = next_state  # 更新状态
        
        print(f'Episode {episode}, Total Reward: {total_reward}')  # 打印每轮的总奖励

# 创建环境
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 实例化Actor和Critic
actor = Actor(state_size, action_size)
critic = Critic(state_size)
# 定义优化器
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

# 开始训练
train(env, actor, critic, actor_optimizer, critic_optimizer, episodes=1000)
