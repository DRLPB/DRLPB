import numpy as np
import torch as T
from networks import DDQN
from memory import ReplayBuffer
from sklearn.preprocessing import StandardScaler


class Agent(object):
    def __init__(self, lr, input_dims, n_actions,epsilon, batch_size,env, eps_dec,
                 capacity=1000000,  f1_dims = 512, f2_dims=256,
                 replace=1000, gamma=0.99,network_name='_eval'): 
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_min = 0.01
        self.epsilon = epsilon
        self.memory = ReplayBuffer(capacity, input_dims,n_actions)
        self.eps_dec = eps_dec
        self.replace = replace
        self.update_cntr = 0
        self.scaler = self._get_scaler(env)

        # DDQN Evaluate network
        self.eval = DDQN(lr=lr, input_dims=self.input_dims,n_actions=self.n_actions,f1_dims=f1_dims, f2_dims=f2_dims,network_name=network_name)
        # DDN Training Network
        self.train = DDQN(lr=lr, input_dims=self.input_dims,n_actions=self.n_actions,f1_dims=f1_dims, f2_dims=f2_dims,network_name=network_name)

    # Normalize the observations
    def pick_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor([obs], dtype=T.float).to(self.eval.device)
            actions = self.train.forward(state)
            action = T.argmax(actions).item()
        else:
            action = self.env.sample_action()

        return action

# Normalizing states 
    def _get_scaler(self, env):
        states = []
        for _ in range(self.env.n_steps):
            action = self.env.sample_action()
            state_, reward, done, _ = self.env.step(action)
            states.append(state_)
            if done: break
        scaler = StandardScaler()
        scaler.fit(states)
        return scaler

    def store_transition(self, state, action, reward, state_, done):
        
        self.memory.store_transition(state, action, reward,state_,done)


    def save(self):
        self.eval.save()
        self.train.save()

    def load(self):
        self.eval.load()
        self.train.load()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.eval.device)
        actions = T.tensor(actions, dtype=T.int64).to(self.eval.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.eval.device)
        states_ =T.tensor(states_, dtype=T.float).to(self.eval.device)
        done = T.tensor(done, dtype=T.bool).to(self.eval.device)

        self.train.optimizer.zero_grad()

        indices = np.arange(self.batch_size)
        q_pred = (self.train.forward(states) * actions).sum(dim=1)
        q_next = self.eval.forward(states_)
        q_train = self.train.forward(states_)

        max_action = T.argmax(q_train,dim=1)
        q_next[done] = 0.0

        y = rewards + self.gamma*q_next[indices,max_action]

        loss = self.train.loss(y,q_pred).to(self.eval.device)
        loss.backward()

       

        self.train.optimizer.step()



        self.update_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        

