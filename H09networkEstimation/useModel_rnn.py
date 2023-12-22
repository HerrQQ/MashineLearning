import torch
from torch import nn
import os

from rnnForYawrate import MyRNN
from rnnForYawrate import CustomDataset_RNN
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# main process        
if __name__ == "__main__":

    # current_path = os.getcwd()
    # print("当前工作路径:", current_path)
    # new_path = current_path
    # os.chdir(new_path)

    data_file = 'validationData.txt'  
    custom_dataset_inference = CustomDataset_RNN(data_file)
    batch_size=16
    # 数据加载器
    data_loader_inference = DataLoader(custom_dataset_inference, batch_size=batch_size,shuffle=False, drop_last=True,pin_memory=True)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model_f=MyRNN().to(device)
    model_f.load_state_dict(torch.load('VehicalStateperML_RNN_epochs_2000_BS:128.pth'))
    model_f.eval()

    if 0:
    # single data inference    
        #hidden_prev init
        hidden_prev = torch.zeros(1, 1, 32) #
        
        # data_test_1=torch.tensor([0.03987893462181091,13.782072067260742,0.21904867887496948,0.8140702247619629,0.14955520629882812,
        # 0.0,0.01074754074215889,8.803881645202637,-0.25467321276664734])
        
        # data_test_2=torch.tensor([0.038624390959739685,13.786596298217773,0.2508492171764374,0.8024824857711792,0.15127170085906982,
        # 0.0,0.010829867795109749,8.872920989990234,-0.3033924400806427])
        
        # data_test_3=torch.tensor([0.03859284520149231,13.785704612731934,0.15484023094177246,0.8001524209976196,0.1524304449558258,
        # 0.0,0.010974112898111343,8.793074607849121,-0.34160029888153076])

        # data_test_4=torch.tensor([0.03933088481426239,13.788601875305176,0.2645440995693207,0.8018520474433899,0.15475332736968994,
        # 0.0,0.01099490374326706,8.803157806396484,-0.4666106402873993])

        # data_test_5=torch.tensor([0.039885520935058594,13.795905113220215,0.16176363825798035,0.7602496147155762,0.15518921613693237,
        # 0.0,0.011025872081518173,8.80613899230957,-0.5674195289611816])

        data_test_1=torch.tensor([0.03987893462181091,13.782072067260742,0.21904867887496948,0.8140702247619629,0.14955520629882812,
        0.01074754074215889])
        
        data_test_2=torch.tensor([0.038624390959739685,13.786596298217773,0.2508492171764374,0.8024824857711792,0.15127170085906982,
        0.010829867795109749])
        
        data_test_3=torch.tensor([0.03859284520149231,13.785704612731934,0.15484023094177246,0.8001524209976196,0.1524304449558258,
        0.010974112898111343])

        data_test_4=torch.tensor([0.03933088481426239,13.788601875305176,0.2645440995693207,0.8018520474433899,0.15475332736968994,
        0.01099490374326706])

        data_test_5=torch.tensor([0.039885520935058594,13.795905113220215,0.16176363825798035,0.7602496147155762,0.15518921613693237,
        0.011025872081518173])
        stacked_data = torch.stack((data_test_1, data_test_2, data_test_3, data_test_4, data_test_5)) #shape (5,9)
        stacked_data=stacked_data.unsqueeze(0)# shape (1,5,9)

        with torch.no_grad():
            stacked_data = stacked_data.to(device)  
            hidden_prev =hidden_prev.to(device)
            output,_ = model_f(stacked_data,hidden_prev)  # 使用模型进行预测
        print(output) 
    else:
        # prepare for print 
        output_file = open("resultRnn.txt", "w")
    #multi data inference model
        log_dir='runs/RNNdemo_usemodel_f1'
        writer = SummaryWriter (log_dir=f'{log_dir}')
        hidden_prev = torch.zeros(1, batch_size, 32)
        num_batches=len(data_loader_inference)
        test_loss,total_loss=0.0,0.0
        loss_flag= False 
        loss_fn = nn.MSELoss()
        list_pre_v=[]
        list_pre_yawrate=[]
        list_y_v=[]
        list_y_yawrate=[]
        with torch.no_grad():
            for batch, (X, y) in enumerate(data_loader_inference):
                X = X.to(device)  # 将数据移到设备上
                #print (X.shape)
                y = y.to(device) 
                #print (y.shape)
                batch_size=X.size(0)
                hidden_prev = torch.zeros(1, batch_size, 32,dtype=X.dtype)
                hidden_prev = hidden_prev.to(device)             
                pred,_ = model_f(X,hidden_prev)
                #print (pred.shape)
                bs,sq,fn=pred.shape
                pred_num=pred.cpu().numpy().copy()
                y_num=y.cpu().numpy().copy()
                for i in range(bs) :
                    for j in range(sq):
                        for q in range(fn-1):          
                            list_pre_yawrate.append(pred_num[i][j][q])
                            list_y_yawrate.append(y_num[i][j][q])

                for i in range(bs) :
                    for j in range(sq):
                        for q in range(1,fn):          
                            list_pre_v.append(pred_num[i][j][q])
                            list_y_v.append(y_num[i][j][q])
                loss_test=loss_fn(pred, y)
                test_loss += loss_test.item()   
                total_loss += loss_test.item()     
                hidden_prev =hidden_prev.to(device)
                cycle2Out=50
                if batch % cycle2Out == 0:# set print frequenz
                    if (False==loss_flag):
                        loss_flag=True
                    else:# set print frequenz
                        #current = (batch + 1) * len(X)
                        #print(f"average test loss of {cycle2Out} batches: {(test_loss/cycle2Out):>7f}  [{current:>5d}/{size:>5d}]")
                        writer.add_scalar(f'average evaluation loss of every {cycle2Out} batches :', test_loss/cycle2Out ,batch+1)
                        test_loss=0.0
        print("_________________________________________")
        print(f"total average evaluation loss {total_loss/num_batches:>8f} \n")
        for i in range(len(list_pre_yawrate)):          
                output_file.write(f'pre_yawrate:{list_pre_yawrate[i]} /ori:{list_y_yawrate[i]} \n')
        for i in range(len(list_pre_v)):          
                output_file.write(f'pre_v:{list_pre_v[i]} /ori:{list_y_v[i]} \n')
        
        
        output_file.close()

# RNN single data inference model
# msg.egoEgoStatus.yawRate: 0.03987893462181091
# msg.egoEgoStatus.linearSpeed: 13.782072067260742
# msg.egoEgoStatus.accerationX: 0.21904867887496948
# msg.egoEgoStatus.accerationY: 0.8140702247619629
# msg.egoEgoStatus.steerWheelAngle: 0.14955520629882812
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.010625587776303291
# msg.nohCtrlOutput.targetStrAngle: 8.803881645202637
# msg.nohCtrlOutput.targetAcceleration: -0.25467321276664734
# msg.egoEgoStatus.yawRate: 0.038624390959739685
# msg.egoEgoStatus.linearSpeed: 13.786596298217773
# msg.egoEgoStatus.accerationX: 0.23861587047576904
# msg.egoEgoStatus.accerationY: 0.8024824857711792
# msg.egoEgoStatus.steerWheelAngle: 0.15127170085906982
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.01074754074215889
# msg.nohCtrlOutput.targetStrAngle: 8.8500394821167
# msg.nohCtrlOutput.targetAcceleration: -0.3033924400806427
# msg.egoEgoStatus.yawRate: 0.03859284520149231
# msg.egoEgoStatus.linearSpeed: 13.785704612731934
# msg.egoEgoStatus.accerationX: 0.2508492171764374
# msg.egoEgoStatus.accerationY: 0.8001524209976196
# msg.egoEgoStatus.steerWheelAngle: 0.1524304449558258
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.010829867795109749
# msg.nohCtrlOutput.targetStrAngle: 8.872920989990234
# msg.nohCtrlOutput.targetAcceleration: -0.34160029888153076
# msg.egoEgoStatus.yawRate: 0.03933088481426239
# msg.egoEgoStatus.linearSpeed: 13.788601875305176
# msg.egoEgoStatus.accerationX: 0.2645440995693207
# msg.egoEgoStatus.accerationY: 0.8018520474433899
# msg.egoEgoStatus.steerWheelAngle: 0.15475332736968994
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.01099490374326706
# msg.nohCtrlOutput.targetStrAngle: 8.803157806396484
# msg.nohCtrlOutput.targetAcceleration: -0.4666106402873993
# msg.egoEgoStatus.yawRate: 0.039885520935058594
# msg.egoEgoStatus.linearSpeed: 13.795905113220215
# msg.egoEgoStatus.accerationX: 0.16176363825798035
# msg.egoEgoStatus.accerationY: 0.7602496147155762
# msg.egoEgoStatus.steerWheelAngle: 0.15518921613693237
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.011025872081518173
# msg.nohCtrlOutput.targetStrAngle: 8.80613899230957
# msg.nohCtrlOutput.targetAcceleration: -0.5674195289611816

#ans:
# msg.egoEgoStatus.yawRate: 0.04009075462818146
# msg.egoEgoStatus.linearSpeed: 13.800976753234863
# msg.egoEgoStatus.accerationX: 0.20156517624855042
# msg.egoEgoStatus.accerationY: 0.7818496227264404
# msg.egoEgoStatus.steerWheelAngle: 0.15529820322990417
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.011033616028726101



