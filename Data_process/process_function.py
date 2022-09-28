import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(root_path)

import LoadData
import numpy as np
import scipy.linalg
import scipy.io
import scipy.sparse
import scipy.signal as signal

# from braindecode.preprocessing import exponential_moving_standardize

from sklearn.model_selection import train_test_split

def train_validation_split(x,y,validation_size,seed = None):
    '''
    Split the training set into a new training set and a validation set
    @author: WenChao Liu
    '''
    if seed:
        np.random.seed(seed)
    label_unique = np.unique(y)
    validation_x = []
    validation_y = []
    train_x = []
    train_y = []
    for label in label_unique:
        index = (y==label)
        label_num = np.sum(index)
        print("class-{}:{}".format(label,label_num))
        class_data_x = x[index]
        class_data_y = y[index]
        rand_order = np.random.permutation(label_num)
        class_data_x,class_data_y = class_data_x[rand_order],class_data_y[rand_order]
        print(class_data_x.shape)
        validation_x.extend(class_data_x[:int(label_num*validation_size)].tolist())
        validation_y.extend(class_data_y[:int(label_num*validation_size)].tolist())
        train_x.extend(class_data_x[int(label_num*validation_size):].tolist())
        train_y.extend(class_data_y[int(label_num*validation_size):].tolist())
    
    validation_x = np.array(validation_x)
    validation_y = np.array(validation_y).reshape(-1)
    
    train_x = np.array(train_x)
    train_y = np.array(train_y).reshape(-1)
    
    print(train_x.shape,train_y.shape)
    print(validation_x.shape,validation_y.shape)
    return train_x,train_y,validation_x,validation_y

def Load_BCIC_2a_raw_data(data_path,ems = False):
    '''
    Load the BCIC 2a data.
    Arg:
        ems(exponential_moving_standardize):This method may led to a better performance when testing on the HGD dataset.
        
    '''
    
    for sub in range(1,10):
        data_name = r'A0{}T.gdf'.format(sub)
        data_loader = LoadData.LoadBCIC(data_name, data_path)
        data = data_loader.get_epochs(tmin=0.5, tmax=4.5)
        train_x = np.array(data['x_data'])[:, :, :1000]
        train_y = np.array(data['y_labels'])
        
        data_name = r'A0{}E.gdf'.format(sub)
        label_name = r'A0{}E.mat'.format(sub)
        data_loader = LoadData.LoadBCIC_E(data_name, label_name, data_path)
        data = data_loader.get_epochs(tmin=0.5, tmax=4.5)
        test_x = np.array(data['x_data'])[:, :, :1000]
        test_y = data['y_labels']
        

        train_x = np.array(train_x)
        train_y = np.array(train_y).reshape(-1)
        # if ems:
        #     train_x = EMstandardize(train_x)
        
        test_x = np.array(test_x)
        test_y = np.array(test_y).reshape(-1)
        # if ems:
        #     test_x = EMstandardize(test_x)
        
        
        print('trian_x:',train_x.shape)
        print('train_y:',train_y.shape)
        
        print('test_x:',test_x.shape)
        print('test_y:',test_y.shape)
        
        SAVE_path = os.path.join(root_path,'Data','BCIC_2a') 

        if not os.path.exists(SAVE_path):
            os.makedirs(SAVE_path)
            
        SAVE_test = os.path.join(SAVE_path,r'sub{}_test'.format(sub))
        SAVE_train = os.path.join(SAVE_path,'sub{}_train'.format(sub))
        
        if not os.path.exists(SAVE_test):
            os.makedirs(SAVE_test)
        if not os.path.exists(SAVE_train):
            os.makedirs(SAVE_train)
            
        scipy.io.savemat(os.path.join(SAVE_train, "Data.mat"), {'x_data': train_x,'y_data': train_y})
        scipy.io.savemat(os.path.join(SAVE_test, "Data.mat"), {'x_data': test_x, 'y_data': test_y})
        print('Saved successfully!')

def Load_BCIC_2b_raw_data(data_path):
    '''
    Load all the 9 subjects data,and save it in the fold of r'./Data'.
    '''
    root_path = os.path.split(current_path)[0]
    
    
    save_path = os.path.join(root_path,r'Data','BCIC_2b')

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    for sub in range(1,10):
        load_raw_data = LoadData.LoadBCIC_2b(data_path,sub)
        save_train_path = os.path.join(save_path,r'sub{}_train'.format(sub))
        save_test_path = os.path.join(save_path,r'sub{}_test').format(sub)
        if not os.path.exists(save_train_path):
            os.makedirs(save_train_path)
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)
        
        train_x,train_y = load_raw_data.get_train_data()
        scipy.io.savemat(os.path.join(save_train_path,'Data.mat'),{'x_data':train_x,'y_data':train_y})
        
        test_x,test_y = load_raw_data.get_test_data()
        scipy.io.savemat(os.path.join(save_test_path,'Data.mat'),{'x_data':test_x,'y_data':test_y})
        
    print('Saved successfully！')

# def EMstandardize(data):
#     new_data = []
#     for i in data:
#         new_data.append(exponential_moving_standardize(i))
#     return  np.array(new_data)

def Load_HGD_raw_data(data_path,ems = True):
    '''
    Load all the data of 14 subjects,and save it in the fold of r'./Data'. 
    ems : exponential_moving_standardize
    '''
    save_path = os.path.join(root_path,r'Data','HGD')
    
    for sub in range(1,15):
        print('sub{}:begin.'.format(sub))
        save_train_path = os.path.join(save_path,'sub{}_train'.format(sub))
        save_test_path = os.path.join(save_path,'sub{}_test'.format(sub))
        load_raw_data = LoadData.LoadHGD(data_path,sub)
        
        if not os.path.exists(save_train_path):
            os.makedirs(save_train_path)
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)
        
        train_x,train_y = load_raw_data.get_train_data()

        # if ems:
        #     train_x = EMstandardize(train_x)

        print(train_x.shape)

        scipy.io.savemat(os.path.join(save_train_path,'Data.mat'),{'x_data':train_x,'y_data':train_y})
        
        train_x,train_y = load_raw_data.get_test_data()
        # if ems:
        #     train_x = EMstandardize(train_x)

        scipy.io.savemat(os.path.join(save_test_path,'Data.mat'),{'x_data':train_x,'y_data':train_y})
        print('sub{}:end.'.format(sub))
    print('Saved successfully！')
    
if __name__ == '__main__':

    # data_path = os.path.join(root_path,'Raw_data','BCICIV_2b_gdf')
    # data_path = os.path.join(root_path,r'Raw_data','high_gamma')

    data_path = os.path.join(root_path,'Raw_data','BCICIV_2a_gdf')
    Load_BCIC_2a_raw_data(data_path,False)

    # Load_HGD_raw_data(True)
    