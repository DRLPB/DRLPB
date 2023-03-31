import pandas as pd

def retrieve_data():
    data = '../DDQN/data/data.csv' 
    df = pd.read_csv(data) 
    df= df.drop(columns=['added_voter_address','added_voter_chain','added_voter_level',
   'added_proposer_address','added_proposer_level','received_transaction_delay','received_proposer_delay',
   'new_proposer_leader_address','new_proposer_leader_level'])

    df = df.dropna()
    data = df.values
    delay= df.loc[:,"received_voter_delay"]
    data.astype(int)

    return data

