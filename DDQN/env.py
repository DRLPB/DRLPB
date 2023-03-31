import numpy as np
from gym.utils import seeding



class env(object):
    def __init__(self, data):
        self.data = data
        self.n_steps, self.n_headers = data.shape
        self.time_step = 0
        self.voter_chains= 300  
        self.mining_rate=0.1
        self.new_mining_rate= 0.1
        self.acceptable_level= True 
        self.mempool_max_size= 500000
        self.mined_voter = 00
        self.mined_proposer=0
        self.total=0
        self.total_blocks = []
        self.m=0
        self.v=0
        self.delay= data
        self.reward =0
        self.throughput = 5000
        self.prism_reward=0
        self.voilation =0
        self.action_space = np.arange(3)
        self.observation_space = np.empty(self.n_headers + 2, dtype=np.float) 
        self.reset()
        self._seed()
        self.voter_avg_delay()

    def voter_avg_delay(self):
        sum_num = 0
        for t in self.data[:,1]:
            sum_num = sum_num + t           

        self.delay = sum_num / len(self.data[:,1])
        return self.delay
    

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.time_step = 0
        self.reward= 0
        self.mined_voter = 0
        self._get_V()
        return self._get_state()

    def step(self,action):
        self.voter_avg_delay()
        assert action in self.action_space
        self._action_set(action)
        self.time_step += 1
        self.mined_proposer= +1
        self.mined_voter= +1
        new_blocks= self.mined_voter + self.mined_proposer
        self.total= new_blocks
        self.total_blocks.append(new_blocks)
        state_ = self._get_state()
        done = self.time_step == self.n_steps - 1
        info = {'Voter': self.new_mining_rate,
                'Proposer': self.reward}
        beta = 0.2 
        return state_, self.reward, done, info

    def sample_action(self):
        if self.delay> 3000:
                self.reward =0
                self.voilation=self.voilation+1
        else:
            action = np.random.choice(self.action_space)
        return action

    def _action_set(self,action):
        s=2
        self.prism_reward = ((300*0.1*self.throughput)/(self.delay*(64000/168)))
        self.reward = self.prism_reward
        if action == 0:
                self.reward= ((self.v*self.m*self.throughput)/(self.delay*(64000/168)))

        elif action == 1:   
                if self.voter_chains> 150 and self.new_mining_rate<0.5:
                    self.voter_chains = self.voter_chains-10
                    self.v= int(self.voter_chains)
                    self.new_mining_rate = (self.mining_rate)/0.01
                    self.m = int(self.new_mining_rate)
                
                self.reward= float((self.v*self.m*self.throughput)/(self.delay*(64000/168)))

        elif action == 2:
                if self.new_mining_rate>0.1 and self.voter_chains<600:
                    self.new_mining_rate = (self.mining_rate)*0.1
                    self.m = int(self.new_mining_rate)
                    self.voter_chains = self.voter_chains+10
                    self.v= int(self.voter_chains)

                self.reward= ((self.v*self.m*self.throughput)/(self.delay*(64000/168)))

        return 
        

    def _get_state(self):
        state = self.observation_space
        state[:2] = self.data[self.time_step]
        return state

    def _get_V(self):
        self.mined_voter = self.data[self.mined_voter][0]
    


