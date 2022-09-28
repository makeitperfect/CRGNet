import sys
import torch 
import os.path
import torch.optim
from torch.utils.data import DataLoader, dataloader
import datetime
import copy

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from Modules import Network
from Modules.Network.spdnet.optimizer import StiefelMetaOptimizer
from  Modules.Network import CRGNet 
from Data_process import EEG_Dataset

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

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
            x,y = x.to(device),y.to(device)
            
            predict = Net(x)
            loss = lossFun(predict,y)
            cor_num += torch.sum(torch.argmax(predict,dim=1)==y,dtype = torch.float32)
            all_loss += loss

        mean_loss = all_loss/len(loader)
        mean_accu = cor_num/len(dataset)
    return mean_loss.detach().cpu(),mean_accu.detach().cpu()


def cv_train(sub,eary_stop_epoch=200,net_name = 'CRGNet',k:int = 10,initial_path = None,continue_train = False):
    
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
    
    # Set the random seed.
    seed = 20210908
    data_seed = 20210723
    torch.manual_seed(seed)
    
    # Using the eary stoping to test the network.
    save_path = os.path.join(rootPath,r'Saved_files','trained_model','KFCV_ES',net_name,'sub_{}'.format(sub))
    
    # file_name = 'CRGNet_KFCV_sub{}.pth'.format(sub)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_dir = os.path.abspath(os.path.join(rootPath,'Data','BCIC_2a')) 
    
    print(data_dir)
    dataset_loader = EEG_Dataset.get_CV_EEG_data(sub,data_dir,k,0.2,data_seed,all_session=True)
    # Learning rate.
    lr = 0.001
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    all_accu = []
    mean_accu = 0.0
    
    for i,(train_dataset,split_train_dataset,split_validation_dataset,test_dataset) in enumerate(dataset_loader):
        module = Network.__dict__[net_name]
        
        Net = module().to(device)
        if initial_path is not None :
            print('Net initialization.')
            Net.load_state_dict[torch.load(initial_path)]
            print('Sucessfully.')
            
        optimizer = torch.optim.Adam(Net.parameters(),lr=lr)
        optimizer = StiefelMetaOptimizer(optimizer)
  
        file_name = 'CRGNet_KFCV_sub{}_fold_{}.pth'.format(sub,i)
        if os.path.exists(os.path.join(save_path,file_name)) and continue_train:
            print('Recover the state and Continue!')
            print(os.path.join(save_path,file_name))
            Net.load_state_dict(torch.load(os.path.join(save_path,file_name)))

            Net.eval()
            _,test_accu = get_test_information(Net,test_dataset,criterion)
            all_accu.append(test_accu)
            mean_accu += test_accu/k
            continue
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
                x,y = x.to(device),y.to(device)
                
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
                best_Net = copy.deepcopy(Net.state_dict())
                optimizer_state = copy.deepcopy(optimizer.optimizer.state_dict())
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
                x,y = x.to(device),y.to(device)
                predict = Net(x)
                loss = criterion(predict,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            for x,y in split_validation_loader:
                x,y = x.to(device),y.to(device)
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
        
        _,test_accu = get_test_information(Net,test_dataset,criterion)
        
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
    eary_stop_epoch = 50
    
    for i in range(1,10):
        cv_train(i,eary_stop_epoch,net_name)
