import sys
import torch 
import os.path
import torch.optim
from torch.utils.data import DataLoader, dataloader
import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

sys.path.append('.')
sys.path.append('./Modules/Network')
sys.path.append('./Modules/Network/spdnet')

from Modules.Network.spdnet.optimizer import StiefelMetaOptimizer
from  Modules.Network import CRGNet 
from Modules import Network
from Data_process import EEG_Dataset

def ho_train(sub,eary_stop_epoch=200,net_name = 'CRGNet',initial_path = None):
    '''
    Use hold out setting to train Net
    @author:Klein Liu
    '''
    
    # #设置随机数的种子
    seed = 20210908
    data_seed = 20210723
    torch.manual_seed(seed)
    
    if net_name is None:
        print('请选择要使用的模型！')
        return
    save_path = os.path.join(r'.\Saved_files\trianed_model\HOCV',datetime.datetime.now().strftime('%Y_%m_%d') +net_name)
    
    file_name = 'CRGNet_HOCV_sub{}.pth'.format(sub)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # data_dir = r'../Data/BCIC_2a'
    data_dir = os.path.join(rootPath,'Data','BCIC_2a') 
    
    train_dataset,split_train_dataset,split_validation_dataset,test_dataset = EEG_Dataset.get_HO_EEG_data(sub,data_dir,0.2,data_seed)


    train_len,split_train_len,split_validation_len,test_dataset_len = len(train_dataset),len(split_train_dataset),len(split_validation_dataset),len(test_dataset)
    print(train_len,split_train_len,split_validation_len,test_dataset_len)
    
    split_train_dataloader = DataLoader(split_train_dataset,batch_size=16,shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    
    # module = CRGNet.__dict__[net_name]
    module = Network.__dict__[net_name]
    Net = module()
    
    if initial_path is not None :
        print('Net initialization.')
        Net.load_state_dict(torch.load(initial_path))
    
    lr = 0.001
    optimizer = torch.optim.Adam(Net.parameters(),lr=lr)
    optimizer = StiefelMetaOptimizer(optimizer)
    
    best_Net = None
    optimizer_state = None
    best_accu = 0.00
    remaining_epoch = eary_stop_epoch
    mini_loss = None
    # 第一步
    for epoch in range(1000):
        
        Net.train()
        for x,y in split_train_dataloader:
            predict = Net(x) 
            loss = criterion(predict,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        Net.eval()
        x = split_train_dataset.feature
        y = split_train_dataset.label
        train_predict = Net(x).detach()
        train_loss = criterion(train_predict,y)
        train_accu = torch.sum(torch.argmax(train_predict,dim=1)==y,dtype=torch.float32)/len(y)
        
        x = split_validation_dataset.feature
        y = split_validation_dataset.label
        vali_predict = Net(x).detach()
        validation_loss = criterion(vali_predict,y)
        validation_accu = float(torch.sum(torch.argmax(vali_predict,dim=1)==y,dtype=torch.float32)/len(y))
        
        x,y = test_dataset.feature,test_dataset.label
        test_predict = Net(x).detach()
        test_accu = torch.sum(torch.argmax(test_predict,dim=1)==y,dtype = torch.float32)/len(y)
        
        remaining_epoch = remaining_epoch-1
        print('Epoch:{0:3}--TraLoss:{1:.3}--Tracc:{2:.3}--VaLoss:{3:.3}--Vaacc:{4:.3}--MaxVacc:{5:.3}--ramainingEpoch:{6:3}--Teacc:{7:4}'.format(epoch,train_loss,train_accu,validation_loss,validation_accu,best_accu,remaining_epoch,test_accu))
        
        if remaining_epoch <=0:
            break
        if  mini_loss is None or train_loss<mini_loss:
            mini_loss = train_loss
            
        if validation_accu>best_accu:
            best_Net = Net.state_dict()
            optimizer_state = optimizer.optimizer.state_dict()
            remaining_epoch = eary_stop_epoch
            best_accu = validation_accu
            
            
    print('早停开始使用全部数据进行训练')
    
    #第二步
    #首先加载最好上一步中最好的模型的模型
    Net.load_state_dict(best_Net)
    optimizer.optimizer.load_state_dict(optimizer_state)
    
    split_validation_dataloader = DataLoader(split_validation_dataset,batch_size=16,shuffle=True)
    for epoch in range(800):
        Net.train()
        for x,y in split_train_dataloader:
            predict = Net(x)
            loss = criterion(predict,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        for x,y in split_validation_dataloader:
            predict = Net(x)
            loss = criterion(predict,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Net.eval()
        x,y = train_dataset.feature,train_dataset.label
        predict = Net(x).detach()
        train_loss = criterion(predict,y)
        train_accu = torch.sum(torch.argmax(predict,dim = 1)==y,dtype = torch.float32)/len(y)
        
        x = split_validation_dataset.feature
        y = split_validation_dataset.label
        predict = Net(x).detach()
        validation_loss = criterion(predict,y)
        validation_accu = torch.sum(torch.argmax(predict,dim=1)==y,dtype=torch.float32)/len(y)
        
        x,y = test_dataset.feature,test_dataset.label
        predict = Net(x).detach()
        test_accu = torch.sum(torch.argmax(predict,dim=1)==y,dtype = torch.float32)/len(y)
          
        print('Epch:{0:3}--TraLoss:{1:.3}--Tracc:{2:.3}--VaLoss:{3:.3}--Vaacc:{4:.3}--Testacc:{5:.3}'.format(epoch,train_loss,train_accu,validation_loss,validation_accu,test_accu))
        
        if validation_loss<mini_loss:
            break
    
    
    
    # 在测试集上检测效果
    x,y = test_dataset.feature,test_dataset.label
    predict = Net(x).detach()
    
    test_accu = torch.sum(torch.argmax(predict,dim=1)==y,dtype = torch.float32)/len(y)
    print('sub:{}--acc:{}'.format(sub,test_accu))
    
    #保存模型
    print(file_name)
    torch.save(Net.state_dict(),os.path.join(save_path,file_name))
    print('模型保存成功')
    
    return test_accu


def ho_test(sub:int,net_name:str,path:str):
    '''
    To test the trained Net
    param:
        net_name: the net name want to load.  
        path: the path of the saved Net.
    @author: WenChao Liu
    '''
    data_dir = r'.\Data'
    test_set = EEG_Dataset.get_test_EEG_data(sub,data_dir)
    module  = CRGNet.__dict__[net_name]
    Net = module()
    Net.load_state_dict(torch.load(path))
    Net.eval()
    test_feature = test_set.feature
    test_label = test_set.label
    predict = Net(test_feature)
    test_accu = torch.sum(torch.argmax(predict,dim=1)==test_label,dtype = torch.float32)/len(test_label)
    print('Accuracy:',test_accu)
    return Net

 
def get_test_information(Net,dataset,lossFun,batch_size = 16):
    '''
    Test the Net in the given dataset.
    Arg:
        Net(Module): The Network want to be test.
        dataset(EEG_Dataset): The dataset used to test the Network.
        lossFun:The loss function.
        batch_size: The size of data loaded to the memory. 
    ''' 
    loader = DataLoader(dataset,batch_size = batch_size)
    cor_num = 0.0
    all_loss = 0.0
    with torch.no_grad():
        for x,y in loader:
            predict = Net(x)
            loss = lossFun(predict,y)
            cor_num += torch.sum(torch.argmax(predict,dim=1)==y,dtype = torch.float32)
            all_loss += loss

        mean_loss = all_loss/len(loader)
        mean_accu = cor_num/len(dataset)
    return mean_loss,mean_accu


def cv_train(sub,eary_stop_epoch=200,net_name = 'CRGNet',k:int = 10,initial_path = None):
    '''
    To train the network in KFCV.
    Arg:
        sub:The subject number.
        eary_stop_epoch:The maximun epoch number to maintain that the validation accuracy  has not been increased.
        net_name: The name of the net_work.
        k:The fold of the cross validation.
        initial_path:The path of the initialization file.
    @Author:WenChao Liu
    '''
    #Set the random seed.
    seed = 20210908
    data_seed = 20210723
    torch.manual_seed(seed)
    
    save_path = os.path.join(r'.\Saved_files\trianed_model\KFCV',net_name+'\sub_{}'.format(sub))
    
    #file_name = 'CRGNet_KFCV_sub{}.pth'.format(sub)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)



    data_dir = os.path.abspath(r'.\Data')  
    print(data_dir)
    dataset_loader = EEG_Dataset.get_CV_EEG_data(sub,data_dir,k,0.2,data_seed)
    # Learning rate.
    lr = 0.001
    
    criterion = torch.nn.CrossEntropyLoss()
    all_accu = []
    mean_accu = 0.0
    
    for i,(train_dataset,split_train_dataset,split_validation_dataset,test_dataset) in enumerate(dataset_loader):
        module = CRGNet.__dict__[net_name]
        Net = module()
        if initial_path is not None :
            print('Net initialization.')
            Net.load_state_dict[torch.load(initial_path)]
            print('Sucessfully.')
            
        optimizer = torch.optim.Adam(Net.parameters(),lr=lr)
        optimizer = StiefelMetaOptimizer(optimizer)
  
        file_name = 'CRGNet_KFCV_sub{}_fold_{}.pth'.format(sub,i)
        best_Net = None
        optimizer_state = None
        best_accu = 0.0
        remaining_epoch = eary_stop_epoch
        mini_loss = None
        split_train_loader = DataLoader(split_train_dataset,batch_size=16,shuffle=True)
        print('Sub{}_Fold{} begin training!'.format(sub,i))
        for epoch in range(1000):
            Net.train()
            for x,y in split_train_loader:
                predict = Net(x)
                loss = criterion(predict,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            Net.eval()
            split_train_loss,split_train_accu = get_test_information(Net,split_train_dataset,criterion)
            
        
            split_validation_loss,split_validation_accu = get_test_information(Net,split_validation_dataset,criterion)
            
            
            test_loss,test_accu = get_test_information(Net,test_dataset,criterion)
            
            remaining_epoch -=1
            if remaining_epoch <=0:
                break
            
            if  mini_loss is None or split_train_loss<mini_loss:
                mini_loss = split_train_loss
                
            if split_validation_accu>best_accu:
                best_Net = Net.state_dict()
                optimizer_state = optimizer.optimizer.state_dict()
                remaining_epoch = eary_stop_epoch
                best_accu = split_validation_accu
            
            print('Epoch:{0:3}--SplTraLoss:{1:.3}--SplTracc:{2:.3}--VaLoss:{3:.3}--Vaacc:{4:.3}--MaxVacc:{5:.3}--ramainingEpoch:{6:3}--Teacc:{7:4}'.format(epoch,split_train_loss,split_train_accu,split_validation_loss,split_validation_accu,best_accu,remaining_epoch,test_accu))
    
        print('Eary stopping and use all data to retrain the Net!')    
        Net.load_state_dict(best_Net)
        optimizer.optimizer.load_state_dict(optimizer_state)
        split_validation_loader = DataLoader(split_validation_dataset,batch_size=16,shuffle=True)
        
        for epoch in range(800):
            Net.train()
            for x,y in split_train_loader:
                predict = Net(x)
                loss = criterion(predict,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            for x,y in split_validation_loader:
                predict = Net(x)
                loss = criterion(predict,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            Net.eval()
            train_loss,train_accu = get_test_information(Net,train_dataset,criterion)
            
            split_validation_loss,split_validation_accu = get_test_information(Net,split_validation_dataset,criterion)

  
            _,test_accu = get_test_information(Net,test_dataset,criterion)


            print('Epch:{0:3}--TraLoss:{1:.3}--Tracc:{2:.3}--VaLoss:{3:.3}--Vaacc:{4:.3}--Testacc:{5:.3}'.format(epoch,train_loss,train_accu,split_validation_loss,split_validation_accu,test_accu))
            
            if split_validation_loss<mini_loss:
                break
        
        x,y = test_dataset.feature,test_dataset.label
        predict = Net(x).detach()
        test_accu = torch.sum(torch.argmax(predict,dim=1)==y,dtype=torch.float32)/len(y)
        info_path = os.path.join(save_path,r'fold_{}_info.txt'.format(i))
        f = open(info_path,mode='w')
        f.write('Accuracy:{}'.format(test_accu))
        f.close()
        print('Fold:{}---Accu:{}'.format(i,test_accu))
        all_accu.append(test_accu)
        mean_accu += test_accu/k
        # Save the Net
        torch.save(Net.state_dict(),os.path.join(save_path,file_name))
        
    print('All_accu:{},Average accuracy:{}'.format(all_accu,mean_accu))
    
    info_path = os.path.join(save_path,r'info.txt')
    f = open(info_path,mode='w')
    f.write('All_accu:{}\nAverage accuracy:{}'.format(all_accu,mean_accu))
    f.close()

     
if __name__ == '__main__':
    net_name = 'CRGNet_BCIC_2a'                                                 
    # # net_name = 'SPDNet_EEG_D64_f_55'
    eary_stop_epoch = 100
    # ho_train(2,eary_stop_epoch,net_name)
    all_accu = []
    sum_acc = 0
    for sub in range(3,4):
        print('sub{} begin training'.format(sub+1))
        # acc = ho_train(sub+1,100,True)
        acc = ho_train(sub+1,eary_stop_epoch,net_name)
        all_accu.append(float(acc))
        sum_acc = sum_acc+acc
    print(all_accu)
    print('ave accu is:',sum_acc/9)
    
    # for i in range(2,10):
        
    #     cv_train(i,eary_stop_epoch,net_name)
    # for sp_num in range()
   
    