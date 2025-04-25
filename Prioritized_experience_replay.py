"""
@author Michele Carletti
Implement a Prioritized Experience Replay (PER) buffer for DQN-based reinforcement learning
"""

import numpy as np

class PER:
    def __init__(self, capacity, alpha=0.6):
        """ Prioritized Experience Replay (PER)\n Buffer with priority"""
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha
    
    def add(self, state, action, reward, next_state, done):
        """ Add an experience to the buffer with the corresponding priority value"""
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """ Sample batch_size elements from the buffer.\n Probabilities are derived from priorities"""
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:self.position])
        scaled_priorities = priorities ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indicies, priorities):
        """ Update experiences priorities"""
        for idx, priority in zip(indicies, priorities):
            self.priorities[idx] = priority


def test_PER():

    memory = PER(10)

    for i in range(10):
        state = np.random.rand()
        action = np.random.randint(0, 4)
        reward = np.random.rand()
        next_state = np.random.rand()
        done = np.random.randint(0, 2)

        memory.add(state, action, reward, next_state, done)
    
    res = memory.sample(2)
    print(f"Sampling result: {res}")



if __name__ == "__main__":
    test_PER()
