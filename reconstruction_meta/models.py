import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import math

class DataConsistencyLayer(nn.Module):

    def __init__(self,us_mask):
        
        super(DataConsistencyLayer,self).__init__()

        self.us_mask = us_mask 

    def forward(self,predicted_img_,us_kspace_):

        nslices = predicted_img_.shape[1] # nslices to do dc.
        updated_imgs = []
        #print (nslices)
 
        for ii in range(nslices):

            # us_kspace     = us_kspace[:,0,:,:]
            predicted_img = predicted_img_[:,ii,:,:]
            us_kspace = us_kspace_[:,ii,:,:,:]
            
            kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
             
            #print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.us_mask.shape)
            #torch.Size([4, 1, 256, 256, 2]) torch.Size([4, 256, 256]) torch.Size([4, 256, 256, 2]) torch.Size([1, 256, 256, 1])
            #print (self.us_mask.dtype,us_kspace.dtype)
            updated_kspace1  = self.us_mask * us_kspace 
            updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img
            #print("updated_kspace1 shape: ",updated_kspace1.shape," updated_kspace2 shape: ",updated_kspace2.shape)
            #updated_kspace1 shape:  torch.Size([4, 1, 256, 256, 2])  updated_kspace2 shape:  torch.Size([4, 256, 256, 2])
            updated_kspace   = updated_kspace1 + updated_kspace2

            updated_img    = torch.ifft(updated_kspace,2,True) 
            
            #update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
            update_img_abs = updated_img[:,:,:,0]
            
            updated_imgs.append(update_img_abs)

        if not nslices == 1:
            update_img_abs = torch.stack(updated_imgs, dim=1)
            return update_img_abs.float()

        return update_img_abs.unsqueeze(1).float()

class StudentNet(nn.Module):
    
    def __init__(self):
        super(StudentNet, self).__init__()

        self.weights = {'fc1':[32,1,3,3],
                        'fc2':[32,32,3,3],
                        'fc3':[32,32,3,3],
                        'fc4':[32,32,3,3],
                        'fc5':[1,32,3,3]}
        
        self.fc1 = nn.Linear(1,np.prod(self.weights['fc1']))
        self.fc2 = nn.Linear(1,np.prod(self.weights['fc2']))
        self.fc3 = nn.Linear(1,np.prod(self.weights['fc3']))
        self.fc4 = nn.Linear(1,np.prod(self.weights['fc4']))
        self.fc5 = nn.Linear(1,np.prod(self.weights['fc5']))

    def forward(self, x, cxt):

        conv_weight = self.fc1(cxt)
        conv_weight = torch.reshape(conv_weight,self.weights['fc1'])
        x1 = F.relu(F.conv2d(x, conv_weight, bias=None, stride=1, padding=1))

        conv_weight = self.fc2(cxt)
        conv_weight = torch.reshape(conv_weight,self.weights['fc2'])
        x2 = F.relu(F.conv2d(x1, conv_weight, bias=None, stride=1, padding=1))

        conv_weight = self.fc3(cxt)
        conv_weight = torch.reshape(conv_weight,self.weights['fc3'])
        x3 = F.relu(F.conv2d(x2, conv_weight, bias=None, stride=1, padding=1))

        conv_weight = self.fc4(cxt)
        conv_weight = torch.reshape(conv_weight,self.weights['fc4'])
        x4 = F.relu(F.conv2d(x3, conv_weight, bias=None, stride=1, padding=1))

        conv_weight = self.fc5(cxt)
        conv_weight = torch.reshape(conv_weight,self.weights['fc5'])
        x5 = F.conv2d(x4, conv_weight, bias=None, stride=1, padding=1)
       
        return x1,x2,x3,x3,x4,x5


class DCStudentNet(nn.Module):

    def __init__(self,args):

        super(DCStudentNet,self).__init__()

        self.cascade1 = StudentNet()
        self.cascade2 = StudentNet()
        self.cascade3 = StudentNet()
        self.cascade4 = StudentNet()
        self.cascade5 = StudentNet()

        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)
        us_mask = us_mask.double()

        self.dc = DataConsistencyLayer(us_mask)

    def forward(self,x,x_k,cxt):

        x1 = self.cascade1(x,cxt) # list of channel outputs 
        x1_dc = self.dc(x1[-1],x_k)

        x2 = self.cascade2(x1_dc,cxt)
        x2_dc = self.dc(x2[-1],x_k)

        x3 = self.cascade3(x2_dc,cxt)
        x3_dc = self.dc(x3[-1],x_k)

        x4 = self.cascade4(x3_dc,cxt)
        x4_dc = self.dc(x4[-1],x_k)

        x5 = self.cascade5(x4_dc,cxt)
        x5_dc = self.dc(x5[-1],x_k)

        return x1,x2,x3,x4,x5,x5_dc


