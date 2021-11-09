import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import os
import scipy.io as sio
from Coil20_model import Coil20_25,Coil20_50,Coil20_75,init_weight
from torchvision import datasets, transforms
import argparse
from torch.utils.data import Dataset
from torchvision.utils import save_image
import shutil
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.init as init
parser = argparse.ArgumentParser(description='PreTrain small dataset')
parser.add_argument('--v','--version', type=int, default=1, metavar='N')
parser.add_argument('--b','--batch' ,type=int, default=1440, metavar='N')
parser.add_argument('--d','--dataset', type=str)
parser.add_argument('--att','--attention', type=str)
parser.add_argument('--num','--num_class', type=int, default=20,metavar='N')
parser.add_argument('--e','--epoch', type=int, default=1000,metavar='N')
parser.add_argument('--c','--compress',type=int, default=50,metavar='N')
parser.add_argument('--l','--layer',type=int, default=1,metavar='N')
parser.add_argument('--lr',type=int, default=1e-3,metavar='N')
parser.add_argument('--count',type=int, default=1,metavar='N')

args = parser.parse_args()
transform = transforms.Compose([transforms.ToTensor(), ])

data_path='./data/COIL20.mat'

model = Coil20_25().cuda()




model.apply(init_weight)


class Small_Dataset(Dataset):
    def __init__(self, transform, path):
        # Transforms
        self.transfrom = transform

        self.data = sio.loadmat(path)

        self.image_arr = self.data['fea']
        self.label_arr = self.data['gnd']
        self.data_len = self.data['fea'].shape[0]
       
        

    def __getitem__(self, idx):
        img=self.image_arr[idx]
        img=np.reshape(img,(32,32,1))
        img=img.astype(np.float32)
        img=img
        img=self.transfrom(img)

        label=self.label_arr[idx]
        

        return (img, label)

    def __len__(self):
        return self.data_len



train_dataset=Small_Dataset(transform=transform,path=data_path)
trainloader=DataLoader(train_dataset,batch_size=args.b,shuffle=True)


if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')





is_best=0
best_acc=0
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler=optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.95)
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 32, 32)
    return x

criterion=nn.MSELoss(reduction="sum")

for epoch in range(args.e):
    model.train()

    running_loss = 0.0


    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, _ = data

        inputs = inputs.cuda()
        
        optimizer.zero_grad()
        outputs= model(inputs)
       
        loss = criterion(outputs, inputs)

#        print(loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
       

 #   scheduler.step()
    print("epoch : {} , loss: {:.8f}".format(epoch + 1, running_loss / len(trainloader)))
    

torch.save(model.state_dict(),'./para/{}_v{}_{}{}_layer{}.pth'.format(args.d,args.v,args.att,args.c,args.l))



