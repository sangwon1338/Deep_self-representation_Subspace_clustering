import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

def init_weight(m):
    if type(m)==(nn.Conv2d or nn.ConvTranspose2d):
        init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)

class Coil20_25(nn.Module):
    def __init__(self,test=False):
        super(Coil20_25, self).__init__()

        self.encoder = Conv_layer(in_planes=1, out_planes=1, kernel_size=3, stride=2,padding=1)
        
        self.decoder_conv1 =nn.ConvTranspose2d(1, 1, 3, stride=2,padding=1)
        self.Relu= nn.ReLU(True)
               
        
        
        
        self.Coef = nn.Parameter(1.0e-4 * torch.ones((1440, 1440)))
        self.softmax = nn.Softmax(dim=1)
        self.test=test
        self.shape=[]

  

    def forward(self, x):
       self.shape.append(x.shape)
       
       if self.test == True:
            x = self.encoder(x)
            x = self.softmax_attention(x)

            z_conv,z_ssc = self.self_expressive(x)
            x = torch.reshape(z_ssc,x.shape)
            x = self.decoder_conv1(x,output_size=self.shape[0])
            x = self.Relu(x)
            return x,z_conv,z_ssc,self.Coef

       else: 
            x = self.encoder(x)
            x = self.softmax_attention(x)
            x = self.decoder_conv1(x,output_size=self.shape[0])
            x = self.Relu(x)
        
            return  x
    def self_expressive(self,x):
        z_conv = x.view(x.shape[0],-1)
        z_ssc = torch.matmul(self.Coef, z_conv)
        return z_conv,z_ssc  
    def softmax_attention(self,x):
        x_shape = x.shape
        x = x.view(x.shape[0],-1)
        x = self.softmax(x)
        x = torch.reshape(x,x_shape)


        

class Coil20_50(nn.Module):
    def __init__(self,test=False):
        super(Coil20_50, self).__init__()

        self.encoder = Conv_layer(in_planes=1, out_planes=2, kernel_size=3, stride=2,padding=1)
        
        self.decoder_conv1 =nn.ConvTranspose2d(2, 1, 3, stride=2,padding=1)
        self.Relu= nn.ReLU(True)
               
        
        
        
        self.Coef = nn.Parameter(1.0e-4 * torch.ones((1440, 1440)))
        self.softmax = nn.Softmax(dim=1)
        self.test=test
        self.shape=[]

  

    def forward(self, x):
       self.shape.append(x.shape)
       
       if self.test == True:
            x = self.encoder(x)
            x = self.softmax_attention(x)
            z_conv,z_ssc = self.self_expressive(x)
            x = torch.reshape(z_ssc,x.shape)
            x = self.decoder_conv1(x,output_size=self.shape[0])
            x = self.Relu(x)
            return x,z_conv,z_ssc,self.Coef

       else: 
            x = self.encoder(x)
            x = self.softmax_attention(x)
            x = self.decoder_conv1(x,output_size=self.shape[0])
            x = self.Relu(x)
        
            return  x
    def self_expressive(self,x):
        z_conv = x.view(x.shape[0],-1)
        z_ssc = torch.matmul(self.Coef, z_conv)
        return z_conv,z_ssc   
    def softmax_attention(self,x):
        x_shape = x.shape
        x = x.view(x.shape[0],-1)
        x = self.softmax(x)
        x = torch.reshape(x,x_shape)

class Coil20_75(nn.Module):
    def __init__(self,test=False):
        super(Coil20_v1_softmax_encoder75_layer1, self).__init__()

        self.encoder = Conv_layer(in_planes=1, out_planes=3, kernel_size=3, stride=2,padding=1)
        
        self.decoder_conv1 =nn.ConvTranspose2d(3, 1, 3, stride=2,padding=1)
        self.Relu= nn.ReLU(True)
               
        
        
        
        self.Coef = nn.Parameter(1.0e-4 * torch.ones((1440, 1440)))
        self.softmax = nn.Softmax(dim=1)
        self.test=test
        self.shape=[]

  

    def forward(self, x):
       self.shape.append(x.shape)
       
       if self.test == True:
            x = self.encoder(x)
            x = self.softmax_attention(x)

            z_conv,z_ssc = self.self_expressive(x)
            x = torch.reshape(z_ssc,x.shape)
            x = self.decoder_conv1(x,output_size=self.shape[0])
            x = self.Relu(x)
            return x,z_conv,z_ssc,self.Coef

       else: 
            x = self.encoder(x)
            x = self.softmax_attention(x)
            x = self.decoder_conv1(x,output_size=self.shape[0])
            x = self.Relu(x)
        
            return  x
    def self_expressive(self,x):
        z_conv = x.view(x.shape[0],-1)
        z_ssc = torch.matmul(self.Coef, z_conv)
        return z_conv,z_ssc   
    def softmax_attention(self,x):
        x_shape = x.shape
        x = x.view(x.shape[0],-1)
        x = self.softmax(x)
        x = torch.reshape(x,x_shape)


class Coil20_375(nn.Module):
    def __init__(self,test=False):
        super(Coil20_375, self).__init__()

        self.encoder = Conv_layer(in_planes=1, out_planes=15, kernel_size=3, stride=2,padding=1)
        
        self.decoder_conv1 =nn.ConvTranspose2d(15, 1, 3, stride=2,padding=1)
        self.Relu= nn.ReLU(True)
               
        
        
        
        self.Coef = nn.Parameter(1.0e-4 * torch.ones((1440, 1440)))
        self.softmax = nn.Softmax(dim=1)
        self.test=test
        self.shape=[]

  

    def forward(self, x):
       self.shape.append(x.shape)
       
       if self.test == True:
            x = self.encoder(x)
            x = self.softmax_attention(x)

            z_conv,z_ssc = self.self_expressive(x)
            x = torch.reshape(z_ssc,x.shape)
            x = self.decoder_conv1(x,output_size=self.shape[0])
            x = self.Relu(x)
            return x,z_conv,z_ssc,self.Coef

       else: 
            x = self.encoder(x)
            x = self.softmax_attention(x)
            x = self.decoder_conv1(x,output_size=self.shape[0])
            x = self.Relu(x)
        
            return  x
    def self_expressive(self,x):
        z_conv = x.view(x.shape[0],-1)
        z_ssc = torch.matmul(self.Coef, z_conv)
        return z_conv,z_ssc   
    def softmax_attention(self,x):
        x_shape = x.shape
        x = x.view(x.shape[0],-1)
        x = self.softmax(x)
        x = torch.reshape(x,x_shape)