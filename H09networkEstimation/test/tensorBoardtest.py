import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 定义问题和数据
X = np.linspace(-np.pi, np.pi, 100)  # 100个时间步
Y = np.sin(X)
X = torch.tensor(X, dtype=torch.float32).view(1,-1, 1)# (100,1)
Y = torch.tensor(Y, dtype=torch.float32).view(1,-1, 1)#(100,1) no batch_size

# 划分训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2. 定义模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) 
        
    def forward(self, x):
        out, _ = self.rnn(x)# (1, 100, hidden_size)
        out = out[:, -1, :]#( (1,  hidden_size))
        return self.fc(out) #(1,1,)

# 3. 设置超参数范围
hidden_sizes = [10, 20, 30]
num_layers = [1, 2]
learning_rates = [0.01, 0.001]

# 初始化TensorBoard
writer = SummaryWriter()

# 4. 训练模型并进行验证
for i, hidden_size in enumerate(hidden_sizes):
    for j, num_layer in enumerate(num_layers):
        for k, learning_rate in enumerate(learning_rates):
            model = RNNModel(1, hidden_size, num_layer)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # 训练模型
            for epoch in range(100):
                model.train()
                outputs = model(X_train)
                loss = criterion(outputs, Y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 在验证集上评估
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, Y_val)
                
                # 使用TensorBoard记录性能指标
                writer.add_scalar('Loss/train', loss.item(), epoch + i*100 + j*1000 + k*10000)
                writer.add_scalar('Loss/val', val_loss.item(), epoch + i*100 + j*1000 + k*10000)
                writer.add_scalar('Hyperparameters/hidden_size', hidden_size, epoch + i*100 + j*1000 + k*10000)
                writer.add_scalar('Hyperparameters/num_layer', num_layer, epoch + i*100 + j*1000 + k*10000)
                writer.add_scalar('Hyperparameters/learning_rate', learning_rate, epoch + i*100 + j*1000 + k*10000)

writer.close()
