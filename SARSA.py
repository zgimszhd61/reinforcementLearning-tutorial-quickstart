import numpy as np
import random

# 环境参数
num_rows = 4  # 网格的行数
num_cols = 4  # 网格的列数
num_states = num_rows * num_cols  # 总状态数
num_actions = 4  # 可能的动作数：上、下、左、右

# 初始化Q表
Q = np.zeros((num_states, num_actions))

# 动作到坐标变换
action_to_delta = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

# 转换状态和坐标
def state_to_xy(state):
    return (state // num_cols, state % num_cols)

def xy_to_state(x, y):
    return x * num_cols + y

# 奖励函数
def reward(state):
    if state == num_states - 1:
        return 100  # 到达终点的奖励
    else:
        return -1  # 每步移动的代价

# 策略：epsilon-贪婪策略
def policy(state, Q, epsilon=0.1):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)  # 探索
    else:
        return np.argmax(Q[state])  # 利用

# SARSA学习算法
def sarsa_learning(Q, alpha=0.5, gamma=0.9, episodes=100):
    for _ in range(episodes):
        state = 0  # 初始状态
        action = policy(state, Q)  # 初始动作
        while state != num_states - 1:
            x, y = state_to_xy(state)
            dx, dy = action_to_delta[action]
            next_x, next_y = x + dx, y + dy

            # 边界检查
            if not (0 <= next_x < num_rows and 0 <= next_y < num_cols):
                next_x, next_y = x, y  # 无效移动，保持原地

            next_state = xy_to_state(next_x, next_y)
            next_action = policy(next_state, Q)  # 下一个动作

            # SARSA更新公式
            Q[state, action] += alpha * (reward(next_state) + gamma * Q[next_state, next_action] - Q[state, action])

            state, action = next_state, next_action  # 更新状态和动作
print(Q)
sarsa_learning(Q)
print("学习后的Q表:")
print(Q)

def get_decision(state):
    """
    根据智能体的当前位置（状态），输出其决策（动作）。

    参数:
        state (int): 智能体的当前状态（位置）的索引。

    返回:
        int: 选定的动作索引，对应于上、下、左、右中的一个。
    """
    # 确保使用全局变量Q
    global Q
    # 应用epsilon-贪婪策略来选择动作
    action = policy(state, Q, epsilon=0.1)
    return action

# 示例使用：假设智能体当前在状态0
current_state = 0
decision = get_decision(current_state)
print(f"在状态 {current_state}，智能体决定采取的动作是 {decision}")

# example_states = [0, 5, 15]

# # # 打印每个状态的决策
# # for state in example_states:
# #     decision = get_decision(state)
# #     print(f"在状态 {state}，智能体决定采取的动作是 {decision} (0=上, 1=下, 2=左, 3=右)")


def simulate_path(start_state, goal_state):
    current_state = start_state
    path = [current_state]  # 记录整个路径的状态

    while current_state != goal_state:
        action = get_decision(current_state)  # 根据当前状态获取决策动作
        x, y = state_to_xy(current_state)
        dx, dy = action_to_delta[action]
        next_x, next_y = x + dx, y + dy

        # 确保移动有效（不越界）
        if 0 <= next_x < num_rows and 0 <= next_y < num_cols:
            next_state = xy_to_state(next_x, next_y)
        else:
            next_state = current_state  # 无效移动，保持原地

        print(f"从状态 {current_state} 采取动作 {action} 移动到状态 {next_state}")
        current_state = next_state
        path.append(current_state)
        if current_state == goal_state:
            print("已到达目标！")
            break

    return path

def plot_path(path):
    grid = np.zeros((num_rows, num_cols))
    for state in path:
        x, y = state_to_xy(state)
        grid[x, y] = 1
    plt.imshow(grid, cmap='hot')
    plt.title("智能体移动路径")
    plt.show()

# 运行示例
path = simulate_path(0, 15)
plot_path(path)


#########

import matplotlib.pyplot as plt
import numpy as np

def plot_path_with_labels(path):
    grid = np.zeros((num_rows, num_cols))
    fig, ax = plt.subplots()

    for state in path:
        x, y = state_to_xy(state)
        grid[x, y] = 1  # 标记路径

    ax.imshow(grid, cmap='hot')  # 显示网格
    ax.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    # 添加状态编号
    for i in range(num_rows):
        for j in range(num_cols):
            ax.text(j, i, xy_to_state(i, j), ha='center', va='center', color='white')

    plt.title("智能体移动路径与网格编号")
    plt.show()

# 使用这个函数来显示带有编号的网格路径
path = simulate_path(0, 15)  # 假设 simulate_path 已经定义
plot_path_with_labels(path)
