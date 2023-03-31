import numpy as np

class ReplayBuffer():
    def __init__(self, capacity, input_dims, n_actions):
        self.max_mem = capacity
        self.mem_cntr = 0
        self.state = np.zeros((capacity, input_dims), dtype=np.float32)
        self.new_state = np.zeros((capacity, input_dims))
        self.action = np.zeros((capacity, n_actions), dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=np.bool)


    def store_transition(self,state, action, reward, state_, done):
        idx = self.mem_cntr % self.max_mem
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.new_state[idx] = state_
        self.terminal[idx] = done

        self.mem_cntr += 1


    def sample_buffer(self, batch_size=32):
        max_mem = min(self.mem_cntr, self.max_mem)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state[batch]
        actions = self.action[batch]
        rewards = self.reward[batch]
        states_ = self.new_state[batch]
        terminal = self.terminal[batch]

        return states, actions, rewards, states_, terminal
