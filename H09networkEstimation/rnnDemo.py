# import torch
# import torch.nn as nn

# class RNNNeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.rnn = nn.RNN(input_size=9, hidden_size=64, num_layers=3, batch_first=True)
#         self.linear = nn.Linear(64, 7)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # Reshape input to (batch_size, sequence_length, input_size)
#         x = x.unsqueeze(dim=0)
        
#         # Pass through RNN layer
#         out, _ = self.rnn(x)
        
#         # Get output from the last time step
#         last_output = out[:, -1, :]
        
#         # Feed into linear layer and apply activation
#         logits = self.linear(last_output)
#         output = self.relu(logits)
        
#         return output

    ######################################################CDSN############################################
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
 
 
class Rnn(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(Rnn, self).__init__()
 
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=3,
            batch_first=True
        )

        self.out = nn.Linear(32, 1)
 
    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state) 
        outs = []
        for time in range(r_out.size(1)):#r_out.size(1) 表示序列的长度。
            #在这个模型中，r_out 的形状是 (batch_size, sequence_length, hidden_size)
            outs.append(self.out(r_out[:, time, :]))# output of every steps
        return torch.stack(outs, dim=1), h_state# h_state is the hidden layer outputs
# 定义一些超参数
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

 
# 选择模型
model = Rnn(INPUT_SIZE)
print(model)
 
# 定义优化器和损失函数
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 
h_state = None # 第一次的时候，暂存为0
 
for step in range(300):# also the test loop
    start, end = step * np.pi, (step+1)*np.pi
 
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
 
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    print("x:{}",x)
    print("y:{}",y)
 
    prediction, h_state = model(x, h_state)# h_state last state in given time series 
    h_state = h_state.data
 
    loss = loss_func(prediction, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
plt.plot(steps, y_np.flatten(), 'r-')
plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
plt.show()