import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import scipy.io as sio
import torchvision.models as models
from sklearn import cluster
from munkres import Munkres
import numpy as np
from PIL import Image
from Coil20_model import Coil20_25,Coil20_50,Coil20_75,init_weight
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
import itertools
parser = argparse.ArgumentParser(description='PyTorch Deep subspace clustering Network')
parser.add_argument('--reg1', type=int, default=150, metavar='N')
parser.add_argument('--reg2', type=int, default=1, metavar='N')
parser.add_argument('--batch' ,type=int, default=1440, metavar='N')
parser.add_argument('--dataset', type=str)
parser.add_argument('--num','--num_class', type=int, default=20,metavar='N')
parser.add_argument('--e','--epoch', type=int, default=1000,metavar='N')
args = parser.parse_args()
transform = transforms.Compose([ transforms.ToTensor(),])
learning_rate = 1e-3
alpha = 0.04
reg1 = args.reg1
reg2 = args.reg2


model = Coil20_25().cuda()

data_path='./data/COIL20.mat'

pretrain_model_PATH='./model_weight/'



model.load_state_dict(torch.load(pretrain_model_PATH))

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
        img=self.transfrom(img)

        label=self.label_arr[idx][0]
        

        return (img, label)

    def __len__(self):
        return self.data_len

train_dataset=Small_Dataset(transform=transform,path=data_path)
trainloader=DataLoader(train_dataset,batch_size=args.batch,shuffle=None)
optimizer = optim.Adam((model.parameters()), lr=learning_rate)
#test function
def best_map(L1, L2):
    # L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def thrC(C, ro):
    if ro < 1:

        N = C.shape[0]

        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate





crition=nn.MSELoss(reduction='sum')
best_acc=0
best_epoch=0
is_best=0
for epoch in tqdm(range(args.e)):

    for i, data in enumerate(trainloader, 0):
        model.train()

        inputs, _ = data
        inputs = inputs.cuda()


        optimizer.zero_grad()
    
        output,z_conv,z_ssc,Coef= model(inputs)
        
        reg_loss=args.reg2*torch.sum(torch.pow(Coef,2))
        ssc_loss=args.reg1*crition(z_conv,z_ssc)
        recon_loss=crition(inputs,output)
        loss = reg_loss+ssc_loss+recon_loss 
        loss.backward()
        
        optimizer.step()
       

    
    with torch.no_grad():
        model.eval()

        for data in trainloader:
            images, labels = data
            images=images.cuda()
            labels = labels.numpy()
           
            _,_,_,Coef=model(images)

            Coef = Coef.cpu().numpy()


            C = thrC(Coef, alpha)
           
            y_x, CKSym_X = post_proC(C, args.num, 12, 8)
            
            missrate_x = err_rate(labels, y_x)
          
            acc = 1 - missrate_x
            if acc>best_acc:
                best_acc=acc
                best_epoch=epoch+1
