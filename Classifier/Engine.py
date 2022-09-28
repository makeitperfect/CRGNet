from sklearn.utils import shuffle
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import sys
import os
import datetime

from zmq import device

cur_path = os.path.abspath(os.path.dirname(__file__))

root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import Modules.Network as Network

'''
感觉无论怎么设计，代码的重用性都不太高。还是要早点完成任务的好。哎，感觉这么做还是没有什么大的创新点！
'''

class Train_network:
    '''
    A Class for easily training the network.
    '''
    def __init__(self,Net_name,lr = 0.001,optim_name = 'AdamW',loss_name = 'CrossEntropyLoss') -> None:
        self.Net_name = Net_name
        self.optim_name = optim_name
        self.sample_loss_name = loss_name
        self.lr = lr
        self.Net = self.load_Net()
        self.optim = self.load_optim()
        self.sample_loss = self.load_loss()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_Net(self,Net_name = None):
        
        '''
        Load the Network.
        '''
        if Net_name is None:
            model = Network.__dict__[self.Net_name]
        else:
            model = Network.__dict__[Net_name]
        Net = model()
        return Net
    
    def load_optim(self,optim_name):
        
        '''
        Load the optimizier.
        '''
        if optim_name is None:
            optim = torch.optim.__dict__[self.optim_name]
        else:
            optim = torch.optim.__dict__[optim_name]
            
        return optim(self.Net.parameters(),lr = self.optim,amsgrad=True,weight_decay=0.0001)

    def load_loss(self,loss_name):
        
        '''
        Load the loss.
        '''
        if loss_name is None:
            loss = torch.nn.__dict__[self.sample_loss_name]
        else:
            loss = torch.nn.__dict__[loss_name]
            
        return loss


    def save_model(self,path):
        '''
        Save the model.
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            print('Make the save path successfully!')
        save_path = os.path.join(path,r'{}_{}.pth'.format(datetime.datetime.now().strftime('%Y_%m_%d'),self.Net_name))
        torch.save(self.Net.state_dict(),path)
        torch.save(self.Net.state_dict(),save_path)


    def get_information(self,dataset,batch_size = 32):
        '''
        Get the accuracy and loss
        '''
        self.Net.eval()
        with torch.no_grad():
            data_len = len(dataset)
            test_data_laoder = DataLoader(dataset,batch_size = batch_size)
            
            accu = 0.0
            for feature,label in test_data_laoder:
                feature = feature.to(self.device)
                label = label.to(self.device)
                pred = self.Net(feature)
                accu += torch.sum(torch.argmax(pred,dim=1)==label,dtype=torch.float32)/data_len
                
        return accu


    def train_one_epoch(self,dataset,batch_size):
        '''
        Trian one epoch,this may need to rewritten for different tasks.
        '''
        self.Net.train()
        data_loader = DataLoader(dataset,batch_size=batch_size,shuffle = True)
        
        for i,(feature,label) in enumerate(data_loader):
            feature = feature.to(self.device)
            label = label.to(self.device)
            
            self.optim.zero_grad()
            pred = self.Net(feature)
    
            loss = self.sample_loss(label,pred)
            loss.backward()
            self.optim.step()
            
        return 
    
    