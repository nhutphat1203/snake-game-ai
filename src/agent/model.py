import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_Net(nn.Module):
    
    def __init__(self, input_size=11, hidden_size=128, output_size=3):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Khởi tạo trọng số (giúp ổn định hơn khi train RL)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)   # Linear output = Q-values
    
    def save(self, file_name='model.pth'):
        model_dir_path = './model'
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        path_file = os.path.join(model_dir_path, file_name)
        torch.save(self.state_dict(), path_file)
        
class Trainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        
        if len(state.shape) == 1: 
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        
        loss.backward()

        self.optimizer.step()