import sys
import torch 
import os.path
import torch.optim
from torch.utils.data import DataLoader
import copy

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

sys.path.append('.')
sys.path.append('./Modules/Network')
sys.path.append('./Modules/Network/spdnet')

from Modules import Network
from Modules.Network.spdnet.optimizer import StiefelMetaOptimizer
from  Modules.Network import CRGNet 
from Data_process import EEG_Dataset

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def get_test_information(Net,dataset,lossFun,batch_size = 16):
    '''
    Test the Net on the given dataset.
    Arg:
        Net(Module): The Network want to be test.
        dataset(EEG_Dataset): The dataset used to test the Network.
        lossFun:The loss function.
        batch_size: The size of data loaded to the memory. 
    ''' 
    loader = DataLoader(dataset,batch_size = batch_size)
    cor_num = 0.0
    all_loss = 0.0
    Net.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            predict = Net(x)
            loss = lossFun(predict,y)
            cor_num += torch.sum(torch.argmax(predict,dim=1)==y,dtype = torch.float32)
            all_loss += loss

        mean_loss = all_loss/len(loader)
        mean_accu = cor_num/len(dataset)
    return mean_loss.detach(),mean_accu.detach()

def ho_train(sub,data_dir,eary_stop_epoch=200,net_name = 'CRGNet',lr = 0.001,batch_size = 16, initial_path = None):

    '''
    Use the hold out setting to train the network, this is the orignal version which  randomly splits the trainning data and trains the network with early stopping. 

    @author:Wenchao Liu
    '''

    # Set the random seed.
    seed = 20210908
    data_seed = 20210723
    torch.manual_seed(seed)
    # seed = None
    # data_seed = None
    # if seed is not None:
    #     torch.manual_seed(seed)
    
    if net_name is None:
        print('Please input the Net name！')
        return

    save_path = os.path.join(rootPath,'Saved_files','trained_model','HOCV_BCIC_2a',net_name)
    
    file_name = '{}_sub{}.pth'.format(net_name,sub)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_dataset,split_train_dataset,split_validation_dataset,test_dataset = EEG_Dataset.get_HO_EEG_data(sub,data_dir,0.2,data_seed)

    train_len,split_train_len,split_validation_len,test_dataset_len = len(train_dataset),len(split_train_dataset),len(split_validation_dataset),len(test_dataset)

    print(train_len,split_train_len,split_validation_len,test_dataset_len)
    
    split_train_dataloader = DataLoader(split_train_dataset,batch_size=batch_size,shuffle=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # module = CRGNet.__dict__[net_name]
    module = Network.__dict__[net_name]
    Net = module().to(device)
    
    if initial_path is not None :
        print('Net initialization.')
        Net.load_state_dict(torch.load(initial_path))
    
    
    optimizer = torch.optim.Adam(Net.parameters(),lr=lr)
    # optimizer = torch.optim.AdamW(Net.parameters(),lr=lr)
    optimizer = StiefelMetaOptimizer(optimizer)
    
    best_Net = None
    optimizer_state = None
    best_accu = 0.00
    remaining_epoch = eary_stop_epoch
    mini_loss = None    
    # First step
    print('Sub {} strats trainning!'.format(sub))
    for epoch in range(1000):
        
        Net.train()
        for x,y in split_train_dataloader:
            x = x.to(device)
            y = y.to(device)
            predict = Net(x) 
            loss = criterion(predict,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Net.eval()
        
        split_train_loss,split_train_accu = get_test_information(Net,split_train_dataset,criterion)
        
        split_validation_loss,split_validation_accu = get_test_information(Net,split_validation_dataset,criterion)
        test_loss,test_accu = get_test_information(Net,test_dataset,criterion)
        remaining_epoch = remaining_epoch-1

        print('Epoch:{0:3}--TraLoss:{1:.3}--Tracc:{2:.3}--VaLoss:{3:.3}--Vaacc:{4:.3}--MaxVacc:{5:.3}--ramainingEpoch:{6:3}--testacc{7:.3}'.format(epoch,split_train_loss,split_train_accu,split_validation_loss,split_validation_accu,best_accu,remaining_epoch,test_accu))
        
        if remaining_epoch <=0:
            break
        if  mini_loss is None or split_train_loss<mini_loss:
            mini_loss = split_train_loss
            
        if split_validation_accu>best_accu:
            best_Net = copy.deepcopy(Net.state_dict())
            optimizer_state = copy.deepcopy(optimizer.optimizer.state_dict())
            # optimizer_state = optimizer.state_dict()
            remaining_epoch = eary_stop_epoch
            best_accu = split_validation_accu
                        
    print('Earyly stopping,and retrain the Net using both the training data and validation data.')
    
    #Second step
    #Firstly,load the best trained model,and retrain the Net using all data.

    Net.load_state_dict(best_Net)
    optimizer.optimizer.load_state_dict(optimizer_state)

    # optimizer.load_state_dict(optimizer_state)

    split_validation_dataloader = DataLoader(split_validation_dataset,batch_size = batch_size,shuffle=True)
    # trian_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    for epoch in range(800):
        Net.train()
        for x,y in split_train_dataloader:
            
            x = x.to(device)
            y = y.to(device)
            predict = Net(x)
            loss = criterion(predict,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        for x,y in split_validation_dataloader:
            x = x.to(device)
            y = y.to(device)
            predict = Net(x)
            loss = criterion(predict,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Net.eval()
        
        train_loss,train_accu = get_test_information(Net,train_dataset,criterion)
    
        split_train_loss,split_train_accu = get_test_information(Net,split_train_dataset,criterion)
        
        split_validation_loss,split_validation_accu = get_test_information(Net,split_validation_dataset,criterion)
        
        test_loss,test_accu = get_test_information(Net,test_dataset,criterion)
        # x,y = test_dataset.feature,test_dataset.label
        # predict = Net(x).detach()
        # test_accu = torch.sum(torch.argmax(predict,dim=1)==y,dtype = torch.float32)/len(y)
          
        print('Epch:{0:3}--TraLoss:{1:.3}--Tracc:{2:.3}--VaLoss:{3:.3}--Vaacc:{4:.3}--Teaccu{5:.3}'.format(epoch,train_loss,train_accu,split_validation_loss,split_validation_accu,test_accu))
        
        if split_validation_loss<mini_loss:
            break    
    
    # Run in the test data.
    Net.eval()
    test_loss,test_accu = get_test_information(Net,test_dataset,criterion)
    
    # x,y = test_dataset.feature,test_dataset.label
    # predict = Net(x).detach()
    
    # test_accu = torch.sum(torch.argmax(predict,dim=1)==y,dtype = torch.float32)/len(y)
    print('sub:{}--loss:{}--acc:{}'.format(sub,test_loss,test_accu))

    #Save the model.
    print(file_name)
    torch.save(Net.state_dict(),os.path.join(save_path,file_name))
    print('The model was saved successfully!')
    
    return test_accu

def hocv_train(sub,data_dir,eary_stop_epoch=200,net_name = 'CRGNet',lr = 0.001,batch_size = 16, initial_path = None):
    '''
    Use hold out setting to train Net. In this version,the k folds cross validation is used to split the trainning data, and using  early stopping strategy to train the network.
    
    @author:Wenchao Liu
    '''
    # Set the random seed.
    seed = None
    data_seed = None
    # torch.manual_seed(seed)

    if seed is not None:
        torch.manual_seed(seed)
    
    if net_name is None:
        print('Please input the Net name！')
        return

    save_path = os.path.join(rootPath,'Saved_files','trained_model','HOCV_BCIC_2a',net_name)
    
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # train_dataset,split_train_dataset,split_validation_dataset,test_dataset = EEG_Dataset.get_HOCV_EEG_data(sub,data_dir,0.2,data_seed)

    models = []
    accuracys = []
    for k,(train_dataset,valid_dataset) in enumerate(EEG_Dataset.get_HOCV_EEG_data(sub,data_dir,5,data_seed)):
        print('sub_{}_fold_{}:'.format(sub,k))

        print(len(train_dataset),len(valid_dataset))
        
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        
        # module = CRGNet.__dict__[net_name]
        module = Network.__dict__[net_name]
        Net = module().to(device)
        
        if initial_path is not None :
            print('Net initialization.')
            Net.load_state_dict(torch.load(initial_path))
        
        optimizer = torch.optim.Adam(Net.parameters(),lr=lr)
        # optimizer = torch.optim.AdamW(Net.parameters(),lr=lr)
        optimizer = StiefelMetaOptimizer(optimizer)
        
        best_Net = None

        # First step
        # print('Sub {} strats trainning!'.format(sub))

        best_Net = None
        optimizer_state = None
        best_accu = 0.00
        remaining_epoch = eary_stop_epoch
        mini_loss = None    
        for epoch in range(1000):
            
            Net.train()
            for x,y in train_dataloader:
                x = x.to(device)
                y = y.to(device)
                predict = Net(x) 
                loss = criterion(predict,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            Net.eval()
            
            train_loss,train_accu = get_test_information(Net,train_dataset,criterion)
            
            valid_loss,valid_accu = get_test_information(Net,valid_dataset,criterion)
            remaining_epoch = remaining_epoch-1

            print('Epoch:{0:3}--TraLoss:{1:.3}--Tracc:{2:.3}--VaLoss:{3:.3}--Vaacc:{4:.3}--MaxVacc:{5:.3}--ramainingEpoch:{6:3}'.format(epoch,train_loss,train_accu,valid_loss,valid_accu,best_accu,remaining_epoch))
            
            if remaining_epoch <=0:
                break
            if  mini_loss is None or train_loss<mini_loss:
                mini_loss = train_loss
                
            if valid_accu>best_accu:
                best_Net = copy.deepcopy(Net.state_dict())
                optimizer_state = copy.deepcopy(optimizer.optimizer.state_dict())
                remaining_epoch = eary_stop_epoch
                best_accu = valid_accu
                        
        print('Earyly stopping,and retrain the Net using both the training data and validation data.')
    
        #Second step
        #Firstly,load the best trained model,and retrain the Net using all data.

        Net.load_state_dict(best_Net)
        optimizer.optimizer.load_state_dict(optimizer_state)

        # optimizer.load_state_dict(optimizer_state)

        valid_dataloader = DataLoader(valid_dataset,batch_size = batch_size,shuffle=True)
        # trian_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        
        for epoch in range(800):
            Net.train()
            for x,y in train_dataloader:
                
                x = x.to(device)
                y = y.to(device)
                predict = Net(x)
                loss = criterion(predict,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            for x,y in valid_dataloader:
                x = x.to(device)
                y = y.to(device)
                predict = Net(x)
                loss = criterion(predict,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            Net.eval()
            
            train_loss,train_accu = get_test_information(Net,train_dataset,criterion)
        
            valid_loss,valid_accu = get_test_information(Net,valid_dataset,criterion)
            
 
            # x,y = test_dataset.feature,test_dataset.label
            # predict = Net(x).detach()
            # test_accu = torch.sum(torch.argmax(predict,dim=1)==y,dtype = torch.float32)/len(y)
            
            print('Epch:{0:3}--TraLoss:{1:.3}--Tracc:{2:.3}--VaLoss:{3:.3}--Vaacc:{4:.3}'.format(epoch,train_loss,train_accu,valid_loss,valid_accu))
            
            if valid_loss<mini_loss:
                break    
        
        #Second step
        #Firstly,load the best trained model,and retrain the Net using all data.
        # Saved_name = ''.format()

        # Net.load_state_dict(best_Net)
        models.append(Net)
        # accuracys.append(valid_accu)
        file_name = '{}_sub{}_fold_{}.pth'.format(net_name,sub,k)
        print(file_name)
        torch.save(Net.state_dict(),os.path.join(save_path,file_name))
        print('The model was saved successfully!')


    # Run in the test data.
    test_dataset = EEG_Dataset.get_test_EEG_data(sub,data_dir)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    cor_num = 0.0
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            predict = None
            for m in models:
                m.eval()
                out = m(x)
                if predict is None:
                    predict = out.detach().cpu()
                else:
                    predict += out.detach().cpu()

            cor_num += torch.sum(torch.argmax(predict,dim=1)==y,dtype = torch.float32)

    test_accu = cor_num/len(test_dataset)
    print('sub:{}--acc:{}'.format(sub,test_accu))
    
    return test_accu

def ho_test(sub:int,data_dir:str,net_name:str,path:str):
    '''
    To test the trained Net.
    param:
        net_name: the net name want to load.  
        path: the path of the saved Net.
    @author: WenChao Liu
    '''
    
    test_dataset = EEG_Dataset.get_test_EEG_data(sub,data_dir)
    module  = CRGNet.__dict__[net_name]
    Net = module()
    Net.load_state_dict(torch.load(path))
    criterion = torch.nn.CrossEntropyLoss()
    Net.eval()
    
    # test_feature = test_set.feature
    # test_label = test_set.label
    # predict = Net(test_feature)
    # test_accu = torch.sum(torch.argmax(predict,dim=1)==test_label,dtype = torch.float32)/len(test_label)

    test_loss,test_accu = get_test_information(Net,test_dataset,criterion) 
    
    print('Loss:{},Accuracy:{}'.format(test_loss,test_accu))
    return Net

if __name__ == '__main__':

    # Config:
    # The operatoins of QR and SVD make the reasults different from different devices,even if fix the random seed. 

    net_name = 'CRGNet_BCIC_2a'                                                   
    data_dir = os.path.join(rootPath,'Data','BCIC_2a') 
    eary_stop_epoch = 100
    batch_size = 6
    lr = 0.001

    all_accu = []
    mean_accu = 0.0

    for i in range(1,10):
        acc = hocv_train(i,data_dir,eary_stop_epoch,net_name,lr,batch_size)
        all_accu.append(acc)
        mean_accu+=acc/9.0

    print('all_accuracy:{},mean_accuracy{}'.format(all_accu,mean_accu))

    # log
    info_path = os.path.join(rootPath,'Saved_files','trained_model','HOCV_BCIC_2a',net_name)
    text_path = os.path.join(info_path,r'info.txt')

    f = open(text_path,mode='w')
    f.write('All_accu:{}\nAverage accuracy:{}'.format(all_accu,mean_accu))
    f.close()

 
