# import tensorflow as tf 
# import  torch
# import  numpy as np
# import  torch.nn as nn
# import  torch.optim as optim
# from    matplotlib import pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torch.optim.lr_scheduler import StepLR 
 
# # # Load and normalize MNIST data 
# # mnist_data = tf.keras.datasets.mnist 
# # (X_train, y_train), (X_test, y_test) = mnist_data.load_data() 
# # X_train, X_test = X_train / 255.0, X_test / 255.0 
 
# # # Define the model 
# # model = tf.keras.models.Sequential([ 
# # tf.keras.layers.Flatten(input_shape=(28, 28)), 
# # tf.keras.layers.Dense(128, activation='relu'), 
# # tf.keras.layers.Dropout(0.2), 
# # tf.keras.layers.Dense(10, activation='softmax') 
# # ]) 
 
# # # Compile the model 
# # model.compile( 
# # optimizer='adam', 
# # loss='sparse_categorical_crossentropy', 
# # metrics=['accuracy'])

# # tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./TBlogs")


# # model.fit(X_train, y_train, epochs=5, callbacks=[tf_callback])


# #############################################################################

# # import numpy as np
# # # Specify a directory for logging data 
# # logdir = "./TB2logs" 
 
# # # Create a file writer to write data to our logdir 
# # file_writer = tf.summary.create_file_writer(logdir) 
 
# # # Loop from 0 to 199 and get the sine value of each number 
# # for i in range(200): 
# #     with file_writer.as_default(): 
# #         tf.summary.scalar('sine wave', np.math.sin(i), step=i)
# #######################################################################################
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()
# import torch 
# import torchvision 
# from torchvision import datasets, transforms 
 
# # Compose a set of transforms to use later on 
# transform = transforms.Compose([ 
#     transforms.ToTensor(),  
#     transforms.Normalize((0.5,), (0.5,)) 
# ]) 
 
# # Load in the MNIST dataset 
# trainset = datasets.MNIST( 
#     'mnist_train',  
#     train=True,  
#     download=True,  
#     transform=transform 
# ) 
 
# # Create a data loader 
# trainloader = torch.utils.data.DataLoader( 
#     trainset,  
#     batch_size=64,  
#     shuffle=True 
# ) 
 
# # Get a pre-trained ResNet18 model 
# model = torchvision.models.resnet18(False) 
 
# # Change the first layer to accept grayscale images 
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
 
# # Get the first batch from the data loader 
# images, labels = next(iter(trainloader)) 
 
# # Write the data to TensorBoard 
# grid = torchvision.utils.make_grid(images) 
# writer.add_image('images', grid, 0) 
# writer.add_graph(model, images) 
# writer.close()
# #################################################################
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.optim as optim

# 设置TensorBoard saving path 
log_dir='runs/fashion_mnist_experiment_3'

# 加载数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)


#set super Para List 
batch_size_list = [5, 10]
lr_list = [.01, .001]

for batch_size in batch_size_list:
    for lr in lr_list:
        # Setting comment and creat writer
        comment = f'batch_size={batch_size} lr={lr}'
        writer = SummaryWriter(log_dir=f'{log_dir}/{comment}')

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last = True)
        # transform = ...: 定义了一个变换序列，将图像转换为PyTorch张量，并对图像进行标准化处理。
        # trainset = torchvision.datasets.FashionMNIST(...): 
        # 加载FashionMNIST数据集，并应用前面定义的变换。

        # 定义一个简单的神经网络
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(28 * 28, 64)
                self.fc2 = nn.Linear(64, 10)

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        net = Net()



        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        # 训练网络
        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 每1000个batch记录一次数据
                if i % 1000 == 999:    
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}")
                    writer.add_scalar('training loss', running_loss / 1000, epoch * len(trainloader) + i)
                    #writer.add_histogram('weight of fc1', net.fc1.weight, epoch * len(trainloader) + i)
                    running_loss = 0.0
            if epoch % 2 == 1 or 0:
                    # 记录每个感兴趣的层的参数
                    for name, param in net.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        print(f'Finished Training in with lr :{lr} BS:{batch_size}')

        # 关闭writer
        writer.close()
