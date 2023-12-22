import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
#from rnnForYawrate import MyRNN
from torch.optim.lr_scheduler import StepLR # add LR Schedule  20 oct.



class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []  

        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = {}
        for line in lines:

            #items = line.strip().split('\n')
            key, value = line.split(': ')
            data[key] = float(value)
            if (key=="msg.nohCtrlOutput.targetAcceleration"):
                self.samples.append(data.copy())
                data.clear()

    def __len__(self):
        return len(self.samples)-1# -1 is special process for my case

    def __getitem__(self, idx):
        sample = self.samples[idx]
        

        features = torch.tensor([
        sample.get('msg.egoEgoStatus.yawRate', 0.0),  # set default value, in some case the dataset could be not printed correctly
        sample.get('msg.egoEgoStatus.linearSpeed', 0.0),
        sample.get('msg.egoEgoStatus.accerationX', 0.0),
        sample.get('msg.egoEgoStatus.accerationY', 0.0),
        sample.get('msg.egoEgoStatus.steerWheelAngle', 0.0),
        sample.get('msg.egoEgoStatus.steerWheelAngleRate', 0.0),
        sample.get('msg.egoEgoStatus.frontWheelAngle', 0.0),
        sample.get('msg.nohCtrlOutput.targetStrAngle', 0.0),
        sample.get('msg.nohCtrlOutput.targetAcceleration', 0.0)
        ])
        sample2 = self.samples[idx+1]# +1 is special process for my case
        labels=torch.tensor([
        sample2.get('msg.egoEgoStatus.yawRate', 0.0),  
        sample2.get('msg.egoEgoStatus.linearSpeed', 0.0),
        sample2.get('msg.egoEgoStatus.accerationX', 0.0),
        sample2.get('msg.egoEgoStatus.accerationY', 0.0),
        sample2.get('msg.egoEgoStatus.steerWheelAngle', 0.0),
        sample2.get('msg.egoEgoStatus.steerWheelAngleRate', 0.0),
        sample2.get('msg.egoEgoStatus.frontWheelAngle', 0.0)
        ])

        return features,labels


# 定义神经网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        #refresh para
        optimizer.step()
        # clear grad
        optimizer.zero_grad()

        if batch % 50 == 0:# set print frequenz
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")



# main process        
if __name__ == "__main__":
    # 数据集
    data_file_learning = 'training.txt' 
    data_file_t = 'test.txt' 
    custom_dataset_training = CustomDataset(data_file_learning)
    custom_dataset_valid = CustomDataset(data_file_t) 

    # para
    learning_rate = 1e-3
    batch_size = 16
    epochs = 5000
    # 数据加载器
    data_loader_training = DataLoader(custom_dataset_training, batch_size=batch_size, shuffle=True,drop_last=True)
    data_loader_valid = DataLoader(custom_dataset_valid, batch_size=batch_size,shuffle=True, drop_last=True)
    # dataset check
    for idx, item in enumerate(data_loader_training):
        print('idx:', idx)
        data, label = item
        print('data:', data)
        print('label:', label)
    for data, label in data_loader_training:
        print(f"Shape of data: {data.shape}")
        print(f"Shape of label: {label.shape} {label.dtype}")
        break
    # check if there are hardwares which contributes to quick computing 
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    if 1:
        model = NeuralNetwork().to(device) 
    #model info debug
    # print(f"Using {device} device")
    # print(f"Model structure: {model}\n\n")
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    # create loss function && optimizer 
    loss_fn = nn.MSELoss()
    if 1:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # create LR schedule
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)  # 设置学习率每x个 epoch 减小为原来的0.x
    # running loop
    for i in range(epochs):
        print(f'_________________Epoch:{i+1}/{epochs}_______________________')
        train_loop(data_loader_training,model,loss_fn,optimizer)
        scheduler.step()
        test_loop(data_loader_training,model,loss_fn)
    # saving model 
    torch.save(model.state_dict(), 'VehicalStateperML.pth')
