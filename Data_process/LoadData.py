import os
import glob
import numpy as np
import scipy.io
import sys
import mne

# from braindecode.preprocessing import exponential_moving_standardize
'''
    This code is part of the fbcsptoolbox,which is a codebase in the github.
    Thanks for the https://fbcsptoolbox.github.io/ to provie this code.
'''

class LoadData:
    def __init__(self, eeg_file_path: str):
        self.eeg_file_path = eeg_file_path
        self.raw_eeg_subject = None

    def load_raw_data_gdf(self, file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self,file_to_load):
        import scipy.io as sio
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self, file_path_extension: str = None):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)

class LoadBCIC(LoadData):
    """Subclass of LoadData for loading BCI Competition IV Dataset 2a"""
    def __init__(self, file_to_load, *args):
        self.stimcodes = ['769', '770', '771', '772']
        # self.epoched_data={}
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        self.fs = None
        super(LoadBCIC, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, baseline=None,reject = False):
        
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        events, event_ids = mne.events_from_annotations(raw_data)
        self.fs = raw_data.info.get('sfreq')
        if reject == True:
            reject_events = mne.pick_events(events,[1])
            reject_oneset = reject_events[:,0]/self.fs
            duration = [4]*len(reject_events)
            descriptions = ['bad trial']*len(reject_events)
            blink_annot = mne.Annotations(reject_oneset,duration,descriptions)
            raw_data.set_annotations(blink_annot)
        
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=True)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        # length = len(self.x_data)
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs
                  }
        return eeg_data
    
class LoadBCIC_2b:
    '''A class to load the test and train data of the BICI IV 2b datast'''
    def __init__(self,path,subject):
        self.subject = subject
        self.path = path
        self.train_name = ['1','2','3']
        self.test_name = ['4','5']
        self.train_stim_code  = ['769','770']
        self.test_stim_code  = ['783']
        self.channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        
    def get_train_data(self):
        data = []
        label = []
        for se in self.train_name:
            data_name = r'B0{}0{}T.gdf'.format(self.subject,se)
            label_name = r'B0{}0{}T.mat'.format(self.subject,se)
            data_path = os.path.join(self.path,data_name)
            label_path = os.path.join(self.path,label_name)
            data_x = self.get_epoch(data_path,True)
            data_y = self.get_label(label_path)
            
            data.extend(data_x)
            label.extend(data_y)
        return np.array(data),np.array(label).reshape(-1)
    
    def get_test_data(self):
        data = []
        label = []
        for se in self.test_name:
            data_name = r'B0{}0{}E.gdf'.format(self.subject,se)
            label_name = r'B0{}0{}E.mat'.format(self.subject,se)
            data_path = os.path.join(self.path,data_name)
            label_path = os.path.join(self.path,label_name)
            data_x = self.get_epoch(data_path,False)
            data_y = self.get_label(label_path)
            
            data.extend(data_x)
            label.extend(data_y)
        return np.array(data),np.array(label).reshape(-1)
            
    
    def get_epoch(self,data_path,isTrain = True):
        raw_data = mne.io.read_raw_gdf(data_path)
        events,events_id =  mne.events_from_annotations(raw_data)
        if isTrain:
            stims = [values for key,values in events_id.items() if key in self.train_stim_code]
        else:
            stims = [values for key,values in events_id.items() if key in self.test_stim_code]
        epochs = mne.Epochs(raw_data,events,stims,tmin =1,tmax = 4,event_repeated='drop',baseline=None,preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        eeg_data = epochs.get_data()*1e6
        return eeg_data[:,:,:750]
    
    def get_label(self,label_path):
        label_info = scipy.io.loadmat(label_path)
        return label_info['classlabel'].reshape(-1)-1

class LoadHGD:
    '''
    A class to load the train data and test dats from raw data. 
    '''
    def __init__(self,path,subject) -> None:
        self.path = path
        self.subject = subject
        self.stims = [1,2,3,4]
        self.channel_names = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4',
                 'CP5', 'CP1', 'CP2', 'CP6',
                 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
                 'CP3', 'CPz', 'CP4',
                 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
                 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h',
                 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h', 'CCP2h', 'CPP1h', 'CPP2h']
    
    def get_remove_channel(self,channel_list):
        channel_names =  ['EEG '+i for i in self.channel_names]
        return  [c for c in channel_list if c not in channel_names ]
         
    def get_epoch(self,file_path):
        raw_eeg = mne.io.read_raw_edf(file_path)
        events,event_id = mne.events_from_annotations(raw_eeg)
        drop_channels = self.get_remove_channel(raw_eeg.info['ch_names'])
        epoch = mne.Epochs(raw_eeg,events,event_id=self.stims,tmin = 0,tmax=4,event_repeated='drop',baseline=None,preload=True,proj=False,reject_by_annotation=True)
        epoch = epoch.drop_channels(drop_channels)
        # epoch = epoch.filter(0.5,100)
        epoch = epoch.resample(250)
        
        x_data = epoch.get_data()
        y_data = epoch.events[:,-1]-min(epoch.events[:,-1])
        
        return x_data,y_data
    
    def get_train_data(self):
        train_path = os.path.join(self.path,'train','{}.edf'.format(self.subject))
        x_data,y_data = self.get_epoch(train_path)
        return np.array(x_data),np.array(y_data).reshape(-1)
    
    def get_test_data(self):
        test_path = os.path.join(self.path,'test','{}.edf'.format(self.subject))
        x_data,y_data = self.get_epoch(test_path)
        return np.array(x_data),np.array(y_data).reshape(-1)
        
class LoadBCIC_E(LoadData):
    """A class to lode the test data of the BICI IV 2a dataset"""
    def __init__(self, file_to_load, lable_name, *args):
        self.stimcodes = ('783')
        # self.epoched_data={}
        self.label_name = lable_name # the path of the test label
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(LoadBCIC_E, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, baseline=None):
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        label_info  = scipy.io.loadmat(os.path.join(self.eeg_file_path,self.label_name))
        #label_info shape:(288, 1)
        self.y_labels = label_info['classlabel'].reshape(-1) -1
        # print(self.y_labels)
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs}
        return eeg_data


if __name__ == '__main__':
    path = r'.\Raw_data\BCICIV_2b_gdf'
    load_raw_data = LoadBCIC_2b(path,5)
    # train_x,train_y = load_raw_data.get_train_data()
    test_x,test_y = load_raw_data.get_test_data()
    print(test_x.shape,test_y.shape)
    # print(train_x.shape,train_y.shape)