
import numpy as np

class GridWorld:
    EMPTY = 0
    OBSTACLE = 1
    AGENT = 'A'
    TARGET = 'T'
    
    def __init__(self, size=5):
        self.size = size
        self.reset()
        
    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i, j] = GridWorld.EMPTY
        
        self.agent_pos = [self.size - 1, 0]
        self.target_pos = [0, self.size - 1]
        
        self.grid[self.agent_pos[0], self.agent_pos[1]] = GridWorld.AGENT
        self.grid[self.target_pos[0], self.target_pos[1]] = GridWorld.TARGET
        self.finished = False
        
    def add_obstacle(self, position):
        if self.grid[position[0], position[1]] == GridWorld.EMPTY:
            self.grid[position[0], position[1]] = GridWorld.OBSTACLE
    
    def move_agent(self, direction):
        if self.finished:
            return 0
        
        new_pos = self.agent_pos.copy()
        if direction == 'up' and self.agent_pos[0] > 0:
            new_pos[0] -= 1
        elif direction == 'down' and self.agent_pos[0] < self.size - 1:
            new_pos[0] += 1
        elif direction == 'left' and self.agent_pos[1] > 0:
            new_pos[1] -= 1
        elif direction == 'right' and self.agent_pos[1] < self.size - 1:
            new_pos[1] += 1
        
        cell_value = self.grid[new_pos[0], new_pos[1]]
        
        if cell_value == GridWorld.OBSTACLE:
            return -1
        elif cell_value == GridWorld.TARGET:
            self.finished = True
            return 1
        
        self.grid[self.agent_pos[0], self.agent_pos[1]] = GridWorld.EMPTY
        self.agent_pos = new_pos
        self.grid[self.agent_pos[0], self.agent_pos[1]] = GridWorld.AGENT
        return -0.01
    
    def __str__(self):
        grid_str = ""
        for row in self.grid:
            grid_str += ' '.join([str(cell) for cell in row]) + '\n'
        return grid_str

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = {}
    
    def get_state(self, env):
        return tuple(env.agent_pos)
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        
        q_values = {action: self.q_table.get((state, action), 0) for action in self.actions}
        max_q_value = max(q_values.values())
        actions_with_max_value = [action for action, value in q_values.items() if value == max_q_value]
        return np.random.choice(actions_with_max_value)
    
    def learn(self, state, action, reward, next_state):
        current_q_value = self.q_table.get((state, action), 0)
        next_q_values = [self.q_table.get((next_state, next_action), 0) for next_action in self.actions]
        new_q_value = current_q_value + self.lr * (reward + self.gamma * max(next_q_values) - current_q_value)
        self.q_table[(state, action)] = new_q_value

def train_agent(agent, env, num_episodes=100, success_threshold=5):
    total_rewards = []
    successful_episodes = 0
    for episode in range(num_episodes):
        state = env.reset()
        state = agent.get_state(env)
        episode_reward = 0
        while not env.finished:
            action = agent.choose_action(state)
            reward = env.move_agent(action)
            next_state = agent.get_state(env)
            agent.learn(state, action, reward, next_state)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
        if episode_reward > 0:
            successful_episodes += 1
        else:
            successful_episodes = 0
        if successful_episodes >= success_threshold:
            x = np.random.randint(1, env.size - 1)
            y = np.random.randint(1, env.size - 1)
            env.add_obstacle([x, y])
            successful_episodes = 0
    return total_rewards
