from re import X
import scipy
from setuptools import sic
import torch.nn as nn
import torch
import os 
import sys
import time
# from thop import profile
from scipy.io import loadmat
import scipy


current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified,SPDVectorize
from utils import Conv2dWithConstraint,LinearWithConstraint


class CRGNet_BCIC_2a(nn.Module):
    '''
    The CRGNet for BCIC IV 2a dataset.
    @Author:WenChao Liu
    '''
    def __init__(self,*args, **kwargs):
        super(CRGNet_BCIC_2a, self).__init__()
        # self.fcon = Sep_fcon(16)
        
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(1,50,kernel_size=(22,1),max_norm=0.5),
            # nn.Conv2d(1,50,kernel_size=(22,1)),
            nn.BatchNorm2d(50),
        )
        
        # self.m = nn.ReLU()
        
        self.temporal_conv_1 = nn.Sequential(
            nn.Conv2d(50,100,kernel_size=(1,15),padding = (0,15//2),groups=50,bias = False),
            nn.BatchNorm2d(100)
        )
        
        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(50,100,kernel_size=(1,35),padding = (0,35//2),groups=50,bias = False),
            nn.BatchNorm2d(100)
        )
        
        self.temporal_conv_3 = nn.Sequential(
            nn.Conv2d(50,100,kernel_size=(1,55),padding = (0,55//2),groups=50,bias = False),
            nn.BatchNorm2d(100)
        )
                 
        self.t1 = SPDTransform(300,64)
        self.r1 = SPDRectified()
        self.tan1 = SPDTangentSpace(64)

       
        self.FC = nn.Linear(2080,4)
        # self.FC = LinearWithConstraint(2080,4,max_norm=0.5)
        
    def forward(self,feature):
        N,C,S = feature.shape
        h = feature.reshape(N,1,C,S)
        h = self.spatial_conv(h)
        # h = self.m(h)
        h1 = self.temporal_conv_1(h)
        h2 = self.temporal_conv_2(h)
        h3 = self.temporal_conv_3(h)
        
        h = torch.cat([h1,h2,h3],1)
        # h = torch.squeeze(h)
        h = rearrange(h,'n c h w -> n c (h w)')
        ht = torch.transpose(h,1,2)
        h = (h@ht)/(S-1)

        h = self.t1(h)
        h = self.tan1(h)
        
        h = self.FC(h)
        return h


class ST_block(nn.Module):
    def __init__(self,spatial_num,channel_num):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            # Conv2dWithConstraint(1,spatial_num,kernel_size=(channel_num,1),max_norm=2),
            nn.Conv2d(1,spatial_num,kernel_size=(channel_num,1)),
            nn.BatchNorm2d(spatial_num),
        )
        
        self.temporal_conv_1 = nn.Sequential(
            nn.Conv2d(spatial_num,spatial_num,kernel_size=(1,15),padding = (0,15//2),groups=spatial_num,bias = False),
            nn.BatchNorm2d(spatial_num)
        )
        
        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(spatial_num,spatial_num,kernel_size=(1,35),padding = (0,35//2),groups=spatial_num,bias = False),
            nn.BatchNorm2d(spatial_num)
        )
        
        self.temporal_conv_3 = nn.Sequential(
            nn.Conv2d(spatial_num,spatial_num,kernel_size=(1,55),padding = (0,55//2),groups=spatial_num,bias = False),
            nn.BatchNorm2d(spatial_num)
        )
        

    def forward(self,input):
        N,C,S = input.shape
        h = input.reshape(N,1,C,S)
        h = self.spatial_conv(h)
        # h = self.m(h)
        h1 = self.temporal_conv_1(h)
        h2 = self.temporal_conv_2(h)
        h3 = self.temporal_conv_3(h)
 
        h = torch.cat([h1,h2,h3],1)
        # h = torch.squeeze(h)
        h = rearrange(h,'n c h w -> n c (h w)')
        return h
        
class CRGNet_HGD(nn.Module):
    '''
    The CRGNet for HGD dataset
    @Author:WenChao Liu
    '''
    def __init__(self,*args, **kwargs):
        super(CRGNet_HGD, self).__init__()
        # self.fcon = Sep_fcon(16)
        self.feature_extract_1 = ST_block(spatial_num=100,channel_num=44)
        self.t1 = SPDTransform(100*3,128)
        self.r1 = SPDRectified()
        self.t2 = SPDTransform(128,64)
        self.tan1 = SPDTangentSpace(128)
      
        # self.FC = LinearWithConstraint(2080,4,max_norm=1)
        self.FC = nn.Sequential(
            nn.Linear(8256,1024),
            nn.Dropout(0.7),
            nn.ReLU(),
            nn.Linear(1024,4)
        ) 
        

    def forward(self,feature):
        N,C,S = feature.shape
        h = self.feature_extract_1(feature)
        ht = torch.transpose(h,1,2)
        h = (h@ht)/(S-1)

        h = self.t1(h)
        # h = self.t2(self.r1(h))
        h = self.tan1(h)
        
        h = self.FC(h)
        return h


class CRGNet_BCIC_2b(nn.Module):
    '''
    The CRGNet for BCIC IV 2b dataset.
    @Author:WenChao Liu
    '''
    def __init__(self,Euci = False,*args, **kwargs):
        super(CRGNet_BCIC_2b, self).__init__()
        self.Euci = Euci
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(1,9,kernel_size=(3,1),max_norm=0.5),
            # nn.Conv2d(1,50,kernel_size=(22,1)),
            nn.BatchNorm2d(9),
        )
        self.temporal_conv_1 = nn.Sequential(
            nn.Conv2d(9,27,kernel_size=(1,15),padding = (0,15//2),groups=9,bias = False),
            nn.BatchNorm2d(27)
        )
        
        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(9,27,kernel_size=(1,25),padding = (0,25//2),groups=9,bias = False),
            nn.BatchNorm2d(27)
        )
        
        self.temporal_conv_3 = nn.Sequential(
            nn.Conv2d(9,27,kernel_size=(1,35),padding = (0,35//2),groups=9,bias = False),
            nn.BatchNorm2d(27)
        )
        
        self.t1 = SPDTransform(81,36)
        self.r1 = SPDRectified()
        self.tan1 = SPDTangentSpace(36,vectorize=False)
        self.vector = SPDVectorize(36)
        
        self.FC = nn.Linear(666,2)

    def feature_extraction(self,feature):
        N,C,S = feature.shape
        h = feature.reshape(N,1,C,S)
        
        hs = self.spatial_conv(h)
        h1 = self.temporal_conv_1(hs)
        h2 = self.temporal_conv_2(hs)
        h3 = self.temporal_conv_3(hs)
        h = torch.cat([h1,h2,h3],1)
        h = rearrange(h,'n c h w -> n c (h w)')
        return hs,h
    
    def riemannian_embedding(self,feature):
        N,_,S = feature.shape
        h = feature
        ht = torch.transpose(h,1,2)
        h = (h@ht)/(S-1)
        c = self.t1(h)
        h = self.tan1(c)
        
        if not self.Euci: 
            embedding_feature = c
        else:
            embedding_feature = h
        return embedding_feature,h
    
    def classifier(self,feature):

        Eh = self.vector(feature)
        h = self.FC(Eh)
        return Eh,h
    
    def forward(self,input):
        
        spatial_feature,feature = self.feature_extraction(input)
        embedding_feature,riemannian_feature = self.riemannian_embedding(feature)
        euclidean_feature,output = self.classifier(riemannian_feature)
        
        return output

class CRGNet(nn.Module):
    '''
    A generic CRGNet structur for Motor imagery classification.
    Arg:
        C:The input data's channel.
        C1:The number of spatial filters.
        temporal_kernels:A list of the temporal convolution kernel size.
        m: The depth of the temporal convolution.
        C2:The dimension of the SPD matrix after bilinear mapping. 
        classes:The category of the EEG data.
         
    @ Author:WenChao Liu
    '''
    def __init__(self,C:int = 22,C1:int = 50,temporal_kernels = [15,35,55],m:int = 2,C2:int = 64,classes:int = 4):
        super().__init__()
        
        self.n  = len(temporal_kernels)
        self.m = m
        self.spatial_conv =nn.Sequential(
            Conv2dWithConstraint(1,C1,kernel_size = (C,1),max_norm= 0.5),
            nn.BatchNorm2d(C1)
            ) 
    
        self.temporal_convs = nn.ModuleList()
        
        for f in temporal_kernels:
            self.temporal_convs.append(nn.Sequential(
                nn.Conv2d(C1,m*C1, kernel_size=(1,f),padding=(0,f//2),groups=C1),
                nn.BatchNorm2d(m*C1)
            ))
        
        self.bilinear_mapping = SPDTransform(self.n*m*C1,C2)
        self.Log_Eig = SPDTangentSpace(C2)
        
        self.FC = nn.Linear((C2+1)*C2//2,classes)
    
    def forward(self,input):
        N,C,T = input.shape
        input = torch.unsqueeze(input,1)
        Xs = self.spatial_conv(input)
        
        Xt = Xs.new(N,self.n,Xs.shape[1]*self.m,1,Xs.shape[3])
        
        for k,temporal_conv in enumerate(self.temporal_convs):
            Xt[:,k,:,:,:] = temporal_conv(Xs)
        
        Xt = Xt.reshape(N,-1,Xt.shape[-1])
        Xt_T = torch.transpose(Xt,1,2)
        Xc = Xt@Xt_T/(T-1)
        
        Xb = self.bilinear_mapping(Xc)
        Xl = self.Log_Eig(Xb)
        Xm = self.FC(Xl)
        
        return Xm

class Simple_Net(nn.Module):
    def __init__(self) -> None:
        super(Simple_Net,self).__init__()
        
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(1,60,kernel_size=(22,1),max_norm=0.5),
            # nn.Conv2d(1,50,kernel_size=(22,1)),
            nn.BatchNorm2d(60),
        )
        self.spatial_conv1d = nn.Sequential(
            nn.Conv1d(22,100,kernel_size=1),
            nn.BatchNorm1d(100),
            nn.Conv1d(100,60,kernel_size=1),
            nn.BatchNorm1d(60)
        )
        self.temporal_conv_1 = nn.Sequential(
            nn.Conv2d(60,120,kernel_size=(1,15),padding = (0,15//2),groups=60,bias = False),
            nn.BatchNorm2d(120)
        )

        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(60,120,kernel_size=(1,35),padding = (0,35//2),groups=60,bias = False),
            nn.BatchNorm2d(120)
        )
        
        self.temporal_conv_3 = nn.Sequential(
            nn.Conv2d(60,120,kernel_size=(1,55),padding = (0,55//2),groups=60,bias = False),
            nn.BatchNorm2d(120)
        )
       
        
        self.t1 = SPDTransform(360,64)
        self.r1 = SPDRectified()
        self.tan1 = SPDTangentSpace(64)

        self.trans= nn.Linear(360,64)
        self.FC = nn.Linear(64,4)
        # self.FC = LinearWithConstraint(2080,4,max_norm=0.5)

    def forward(self,feature):
        N,C,S = feature.shape
        h = feature.reshape(N,1,C,S)
        # h = self.eca(h)
        h = self.spatial_conv(h)
        
        h1 = self.temporal_conv_1(h)
        h2 = self.temporal_conv_2(h)
        h3 = self.temporal_conv_3(h)  
        h = torch.cat([h1,h2,h3],1)
        
        h = rearrange(h,'b c h w -> b h w c')
        h = self.trans(h)
        h = rearrange(h,'b h w c -> b c h w')
        
        h = h**2
        h = reduce(h,' b c h w -> b c',reduction='mean')
        h = torch.log(h)
        
        h = self.FC(h)
        return h
  
if __name__ == '__main__':
    
    
    x = torch.randn(1,44,1000)
    print(x.shape)

    # net = CRGNet_BCIC_2a()
    net = CRGNet_HGD()
    
    y = net(x)
    print(y.shape)
    
    
    
    
    # data_path = r'Data\BCIC_2a'
    # train_data = loadmat(os.path.join(data_path,r'sub1_train/Data.mat'))
    
    
    # feature = train_data['x_data']
    # feature = torch.FloatTensor(feature)[1,:,:][None,:,:]
    # print(feature.shape)
    
    # model_path = r'E:\Project\GitProject\EEG_Codebase\Saved_files\trained_model\HOCV\2021_12_29CRGNet_BCIC_2a\CRGNet_BCIC_2a_sub1.pth'
    
    # state = torch.load(model_path,map_location=torch.device('cpu'))
    
    # net = CRGNet_BCIC_2a()
    # net.load_state_dict(state)
    # net.eval()
    
    # # print(net)
    # begin = time.time()
    # y = net(x)
    # end  = time.time()
    

    # print('run time:',(end-begin)/288)
    
    # total = sum([param.nelement() for param in net.parameters()])
    # print("Number of parameter: %.f" % (total))
    # print(y.shape)