import os
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]

sys.path.append(root_path)
sys.path.append('./Data_process')

from Data_process.process_function import train_validation_split

class my_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,domain = None,domain_label =  False):
        super(my_dataset,self).__init__()
        self.domain_label = domain_label
        self.feature = feature
        self.label = label
        self.domain = domain
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        if self.domain_label:
            return self.feature[index], self.label[index],self.domain[index]
        else:
            return self.feature[index], self.label[index]


def get_test_EEG_data(sub,data_path):
    '''
    Return one subject's test dataset.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
    @author:WenChao Liu 
    '''
    test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))
    test_data = sio.loadmat(test_path)
    test_x = test_data['x_data']
    test_y = test_data['y_data']
    test_x,test_y = torch.FloatTensor(test_x),torch.LongTensor(test_y).reshape(-1)
    test_dataset = my_dataset(test_x,test_y)
    return test_dataset


def get_HO_EEG_data(sub,data_path,validation_size=0.2,data_seed=20210902):
    
    '''
    Return one subject's training dataset,split training dataset and split validation dataset.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
        augment(bool):Data augment.Take care that this operation will change the size in the temporal dimension.
        validation_rate:The percentage of validation data in the data to be divided. 
        data_seed:The random seed for shuffle the data.
    @author:WenChao Liu
    '''
    train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
   
    train_data = sio.loadmat(train_path)
    train_x =  train_data['x_data']
    train_y = train_data['y_data'].reshape(-1)
    print(train_x.shape,train_y.shape)
        
    split_train_x,split_train_y,split_validation_x,split_validation_y = train_validation_split(train_x,train_y,validation_size,seed=data_seed)
    
    train_x,train_y = torch.FloatTensor(train_x),torch.LongTensor(train_y).reshape(-1)
    split_train_x,split_train_y = torch.FloatTensor(split_train_x),torch.LongTensor(split_train_y).reshape(-1)
    split_validation_x,split_validation_y = torch.FloatTensor(split_validation_x),torch.LongTensor(split_validation_y).reshape(-1)
   
    train_dataset = my_dataset(train_x,train_y)
    split_train_dataset = my_dataset(split_train_x,split_train_y)
    split_validation_dataset = my_dataset(split_validation_x,split_validation_y)    
    test_dataset = get_test_EEG_data(sub,data_path)
    
    return train_dataset,split_train_dataset,split_validation_dataset,test_dataset


def get_CV_EEG_data(sub,data_path,k=10,validation_size=0.2,data_seed=20210902,all_session = False):
    '''
    Get the data in KFCV. 
    Arg:
        sub: Subject number.
        data_path: The data  path of all subjects.
        augment(bool):Data augment.Take care that this operation will change the size in the temporal dimension.
        k: K folds cross validation. 
        validation_size:The percentage of validation data in the training data to be divided.
        data_seed:To shuffel the data in the function:train_validation_split.
    Return: A generator to get the kfcv data.
    '''
    path = os.path.join(data_path,'sub{}_train'.format(sub),'Data.mat')
    data = sio.loadmat(path)
    
    data_x = data['x_data']
    data_y = data['y_data'].reshape(-1)
    
    
    
    if all_session:
        session_2_path = os.path.join(data_path,r'sub{}_test'.format(sub),'Data.mat')
        session_2_data = sio.loadmat(session_2_path)
        session_2_x = session_2_data['x_data']
        session_2_y = session_2_data['y_data'].reshape(-1)
        print(data_x.shape)
        print(session_2_x.shape)
        data_x = np.concatenate((data_x,session_2_x))
        data_y = np.concatenate((data_y,session_2_y))
        
    skf = StratifiedKFold(n_splits=k,shuffle=True,random_state= data_seed)
    
    for train_index,test_index in skf.split(data_x,data_y):
        train_x = data_x[train_index]
        train_y = data_y[train_index]
        test_x = data_x[test_index]
        test_y = data_y[test_index]
        print(train_x.shape)
        print(train_y.shape)
        
        split_train_x,split_train_y,split_validation_x,split_validation_y = train_validation_split(train_x,train_y,validation_size,seed=data_seed)
        
        train_x = torch.FloatTensor(train_x)
        train_y = torch.LongTensor(train_y)
        test_x = torch.FloatTensor(test_x)
        test_y = torch.LongTensor(test_y)
        split_train_x = torch.FloatTensor(split_train_x)
        split_train_y = torch.LongTensor(split_train_y)
        split_validation_x = torch.FloatTensor(split_validation_x)
        split_validation_y = torch.LongTensor(split_validation_y)
        
        yield my_dataset(train_x,train_y),my_dataset(split_train_x,split_train_y),my_dataset(split_validation_x,split_validation_y),my_dataset(test_x,test_y)


def get_HOCV_EEG_data(sub,data_path,k=5,data_seed=20210902):
    
    '''
    This version dosen't use early stoping.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
        augment(bool):Data augment.Take care that this operation will change the size in the temporal dimension.
        validation_rate:The percentage of validation data in the data to be divided. 
        data_seed:The random seed for shuffle the data.
    @author:WenChao Liu
    '''
    train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
   
    train_data = sio.loadmat(train_path)
    train_x =  train_data['x_data']
    train_y = train_data['y_data'].reshape(-1)
    print(train_x.shape,train_y.shape)


    skf =   StratifiedKFold(n_splits=k,shuffle=True,random_state= data_seed)
    for split_train_index,split_validation_index in skf.split(train_x,train_y):
        split_train_x = train_x[split_train_index]
        split_train_y = train_y[split_train_index]
        split_validation_x =train_x[split_validation_index]
        split_validation_y = train_y[split_validation_index]

    
        train_x,train_y = torch.FloatTensor(train_x),torch.LongTensor(train_y).reshape(-1)
        split_train_x,split_train_y = torch.FloatTensor(split_train_x),torch.LongTensor(split_train_y).reshape(-1)
        split_validation_x,split_validation_y = torch.FloatTensor(split_validation_x),torch.LongTensor(split_validation_y).reshape(-1)
   
        split_train_dataset = my_dataset(split_train_x,split_train_y)
        split_validation_dataset = my_dataset(split_validation_x,split_validation_y)    
        # test_dataset = get_test_EEG_data(sub,data_path)
    
        yield split_train_dataset,split_validation_dataset



if __name__ == '__main__':
    
    # Test function:get_CV_EEG_data.
    path = os.path.join(root_path,'Data','BCIC_2a')
    
    # tr_,ta_,v_,t_ = get_CSU_selected_sub(1,[3,4,6,7],path)
    
    # tr_,ta_,v_,t_ = get_CSU_EEG_data(1,path,False,True,False,True)
    # source,target_train,target_validation,target_test = get_CSE_data(1,path,False,3,False)
    
    for source,target_train,target_validation,target_test in get_CV_EEG_data(1,path,10,0.2,all_session=True):
    
        print(len(source),len(target_train),len(target_validation),len(target_test))
    
    # print(tr_.feature.shape,ta_.feature.shape,v_.feature.shape,t_.feature.shape)
    # print(tr_.label.shape,ta_.label.shape,v_.label.shape,t_.label.shape)
    # unique_class = torch.unique(v_.label)
    # for i in unique_class:
    #     n = torch.sum(v_.label == i)
    #     print('the {}:{}'.format(i,n))
