from agent_dqn import Agent
from env import env
from data import retrieve_data
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import trange
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDQN Prism")
    parser.add_argument('-load', type=bool, default=False)
    parser.add_argument('-games',type=int, default=1000)
    args = parser.parse_args()
    #  retrieve prism data, load agent, and create the environment
    data = retrieve_data()
    load_agent = args.load
    env = env(data)
    
    
    if load_agent:
        print('Trained Agent Loading...')
        agent = Agent(lr=0.0003, input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                      batch_size=16, epsilon=0.1, env=env, eps_dec=1e-7, replace=1000)
        agent.load()
    else:
        print('Untrained Agent Loading...')
        agent = Agent(lr=0.0003, input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                    batch_size=16, epsilon=1.0, env=env, eps_dec=1e-7, replace=1000)


    
    voter_chains = []
    delay= []
    avg = []
    Episode = []
    Voter_Chains = []
    Mining_Rate = []
    Epsilon = []
    Reward = []
    epsilon_decay=[]
    
    
    # Agent and Environment 
    for i in trange(args.games):
        obs = env.reset()
        done = False
        while not done:
            action = agent.pick_action(obs)
            obs_, reward, done, info = env.step(action)
            if not load_agent:
                agent.store_transition(obs, action, reward, obs_, done)
                agent.learn()
            obs = obs_
       
    
        
        rew = agent.env.reward
        Reward.append(rew)
        v_c = agent.env.voter_chains
        Voter_Chains.append(v_c)
        M_r = env.new_mining_rate
        Mining_Rate.append(M_r)
        epi =i
        Episode.append(epi)
        d= agent.env.delay
        delay.append(d)
        e= agent.epsilon
        epsilon_decay.append(e)
        
        

        print(f'Episode {i}: Gamma {agent.gamma} | Voter Chains  {agent.env.voter_chains}| Mining Rate {env.new_mining_rate} |Epsilon {agent.epsilon:.3f} | Reward {agent.env.reward} | Constraint Voilation {agent.env.voilation} ')
        # plt.plot(i, agent.epsilon, 'go--', linewidth=2, markersize=12) 
    # plt.show()

    avg_reward = np.cumsum(Reward) / np.arange(1, len(Reward) + 1)

    avg_v = np.cumsum(Voter_Chains) / np.arange(1, len(Voter_Chains) + 1)
    avg_miningrate = np.cumsum(Mining_Rate) / np.arange(1, len(Mining_Rate) + 1)

    out_data = {'Episode': Episode,
         'Epsilon': epsilon_decay,
         'Reward': Reward,
         'Voter_Chains': Voter_Chains, 'Mining_Rate': Mining_Rate, 'Avg_Rewards': avg_reward}

    df = pd.DataFrame.from_dict(out_data, orient='index')
    df = df.transpose()
    df.to_csv('output.csv',index=False,header=True, encoding='utf-8')
    


#     smoothing_window = 10
#     fig2 = plt.figure(figsize=(10,5))
#     plt.figure()
#     rewards_smoothed = pd.Series(Reward).rolling(smoothing_window, min_periods=smoothing_window).mean()
# #   y = scipy.stats.norm.cdf(rewards_smoothed)
#     plt.plot(rewards_smoothed)
#     plt.axhline(y=agent.env.prism_reward, color='r', linestyle='-')
#     # plt.plot(agent.env.prism_reward)
#     plt.xlabel("Episode")
#     plt.ylabel(" Reward")
#     plt.title(" Reward over Time".format(smoothing_window))
#     # plt.show()
#     plt.savefig('Reward.png')
    
   
    

 


    
   

    

