import pandas as pd
import numpy as np

output = pd.DataFrame(columns=['added_voter_address',
'added_voter_chain',
'added_voter_level',
'added_proposer_address',
'added_proposer_level',
'received_transaction_delay',
'received_voter_delay',
'received_proposer_delay',
'new_proposer_leader_address',
'new_proposer_leader_level'])

added_voter_address = []
added_voter_chain = []
added_voter_level = []
added_proposer_address = []
added_proposer_level = []
received_transaction_delay = []
received_voter_delay = []
received_proposer_delay = []
new_proposer_leader_address = []
new_proposer_leader_level = []

log_input = open("0.log", "r")
for line in log_input:
    if 'DEBUG - Adding voter block' in line:
        added_voter_address_ = line.split()[5]
        added_voter_chain_ = line.split()[8]
        added_voter_level_ = line.split()[10]
        
        added_voter_address.append(added_voter_address_)
        added_voter_chain.append(added_voter_chain_)
        added_voter_level.append(added_voter_level_)

    elif 'DEBUG - Adding proposer block' in line:
        added_proposer_address_ = line.split()[5]
        added_proposer_level_ = line.split()[8]
        
        added_proposer_address.append(added_proposer_address_)
        added_proposer_level.append(added_proposer_level_)
        
    elif 'DEBUG - Received Transaction block' in line:
        received_transaction_delay_ = (line.split()[5]).split("=")[1]
        received_transaction_delay.append(received_transaction_delay_)
        
    elif 'DEBUG - Received Voter block' in line:
        received_voter_delay_ = (line.split()[5]).split("=")[1]
        received_voter_delay.append(received_voter_delay_)
        
    elif 'DEBUG - Received Proposer block' in line:
        received_proposer_delay_ = (line.split()[5]).split("=")[1]
        received_proposer_delay.append(received_proposer_delay_)
        
    elif 'INFO - New proposer leader selected for level' in line:
        new_proposer_leader_address_ = line.split()[-1]
        new_proposer_leader_level_ = (line.split()[8]).split(":")[0]
        
        new_proposer_leader_address.append(new_proposer_leader_address_)
        new_proposer_leader_level.append(new_proposer_leader_level_)
    

length = max(len(added_voter_address), len(added_voter_chain),len(added_voter_level),
  len(added_proposer_address), len(added_proposer_level), len(received_transaction_delay), 
   len(received_voter_delay), len(received_proposer_delay), len(new_proposer_leader_address), len(new_proposer_leader_level))

added_voter_address.extend([None] * (length-len(added_voter_address)))
added_voter_chain.extend([None] * (length-len(added_voter_chain)))
added_voter_level.extend([None] * (length-len(added_voter_level)))
added_proposer_address.extend([None] * (length-len(added_proposer_address)))
added_proposer_level.extend([None] * (length-len(added_proposer_level)))
received_transaction_delay.extend([None] * (length-len(received_transaction_delay)))
received_voter_delay.extend([None] * (length-len(received_voter_delay)))
received_proposer_delay.extend([None] * (length-len(received_proposer_delay)))
new_proposer_leader_address.extend([None] * (length-len(new_proposer_leader_address)))
new_proposer_leader_level.extend([None] * (length-len(new_proposer_leader_level)))

output['added_voter_address'] =added_voter_address 
output['added_voter_chain'] = added_voter_chain
output['added_voter_level'] = added_voter_level
output['added_proposer_address'] = added_proposer_address
output['added_proposer_level'] = added_proposer_level
output['received_transaction_delay'] = received_transaction_delay
output['received_voter_delay'] = received_voter_delay
output['received_proposer_delay'] =received_proposer_delay
output['new_proposer_leader_address'] = new_proposer_leader_address
output['new_proposer_leader_level'] = new_proposer_leader_level

output.to_csv("./data.csv")
