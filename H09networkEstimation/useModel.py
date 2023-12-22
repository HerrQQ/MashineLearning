import torch
from torch import nn

from trainingModel import NeuralNetwork
from trainingModel import CustomDataset
from rnnForYawrate import MyRNN
from torch.utils.data import Dataset, DataLoader
# main process        
if __name__ == "__main__":


    data_file = 'validationData.txt'  
    custom_dataset_inference = CustomDataset(data_file)

    batch_size=16
    # # 数据加载器
    # data_loader_training = DataLoader(custom_dataset_training, batch_size=batch_size, shuffle=True,drop_last=True)
    data_loader_inference = DataLoader(custom_dataset_inference, batch_size=batch_size,shuffle=True, drop_last=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if 1:
        model_f=NeuralNetwork().to(device)
    else :
        model_f=MyRNN().to(device)
    model_f.load_state_dict(torch.load('VehicalStateperML.pth'))
    model_f.eval()
    # single data inference model
# msg.egoEgoStatus.yawRate: 0.04120563715696335
# msg.egoEgoStatus.linearSpeed: 13.793313980102539
# msg.egoEgoStatus.accerationX: 0.15484023094177246
# msg.egoEgoStatus.accerationY: 0.7746685743331909
# msg.egoEgoStatus.steerWheelAngle: 0.15446069836616516
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.010974112898111343
# msg.nohCtrlOutput.targetStrAngle: 8.793074607849121
# msg.nohCtrlOutput.targetAcceleration: -0.7578315138816833
# msg.egoEgoStatus.yawRate: 0.04082025587558746
# msg.egoEgoStatus.linearSpeed: 13.787456512451172
# msg.egoEgoStatus.accerationX: 0.118187814950943
# msg.egoEgoStatus.accerationY: 0.8204457759857178
# msg.egoEgoStatus.steerWheelAngle: 0.15293440222740173
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.010865672491490841
    if 0:
        data_test=[0.04120563715696335,13.793313980102539,0.15484023094177246,0.7746685743331909,0.15446069836616516,
        0.0,0.010974112898111343,8.793074607849121,-0.7578315138816833]
        with torch.no_grad():
            input_data = torch.tensor(data_test)  # 替换为输入数据
            input_data = input_data.to(device)  
            output = model_f(input_data)  # 使用模型进行预测
        print(output) 
    else:
    # single data inference model
        loss_fn = nn.MSELoss()
        trainingModel.test_loop(data_loader_inference, model_f, loss_fn)




