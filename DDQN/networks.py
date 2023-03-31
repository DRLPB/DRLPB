import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os


class DDQN(nn.Module):
    def __init__(self, lr, input_dims, n_actions,f1_dims, f2_dims,network_name, chkpt_dir="models"):
        super(DDQN, self).__init__()
        self.file = os.path.join(chkpt_dir, network_name+'_ddqn')
        self.f1_dims = f1_dims
        self.f2_dims = f2_dims
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.relu = nn.ReLU()
        self.f1 = nn.Linear(self.input_dims, self.f1_dims)
        self.f2 = nn.Linear(self.f1_dims, self.f2_dims)
        self.q = nn.Linear(self.f2_dims, n_actions)
        self.f_adv = nn.Linear(self.f2_dims, n_actions)
        self.adv = nn.Linear(self.f2_dims, n_actions)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr =lr)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.f1(state)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        q = self.q(x)
        y=self.f_adv(x)
        adv = self.relu(y)

        advAverage = T.mean(adv, dim=1, keepdim=True)
        Q = q + adv - advAverage

        return Q

    def save(self):
        T.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))
