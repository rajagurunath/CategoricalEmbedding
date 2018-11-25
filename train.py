# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 01:45:39 2018

@author: Arunodhaya
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataloader,dataset
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
#from skorch import NeuralNetClassifier
import matplotlib.pyplot as plt
run=wandb.init()
def crt_emb(df):
    df['thal']=df['thal'].replace({3:0,6:1,7:2})
    df['slope']=df['slope'].replace({1:0,2:1,3:2})
    df['exercise_induced_angina']=df['exercise_induced_angina'].replace({1:0,2:1,3:2})
    return df

def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x,x2,x3,x4,y_act = next(data_loader_iter)
    try:
        writer.add_graph(model,input_to_model=(x,x2,x3,x4))
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer
def get_loaders(train_btch_sz,val_btch_sz):
    trn=HealthDataset(X_train,y_train)
    tst=HealthDataset(X_test,y_test)
    trn_loader=dataloader.DataLoader(trn,batch_size=train_btch_sz)
    tst_loader=dataloader.DataLoader(tst,batch_size=val_btch_sz)
    return trn_loader,tst_loader
    
class HeartDisDet(nn.Module):
    """
    7 categorical variables
    with 2,3,4 unique values
    """
    def __init__(self,inp_sz=6,emb_sz=[4,6,8],hdn_sz=[],out_sz=2):
        super(HeartDisDet,self).__init__()
        #inp_sz=np.sum(emb_sz)
        self.emb_2=[nn.Embedding(2,emb_sz[0]) for i in range(0,3)]
        self.emb_3=[nn.Embedding(3,emb_sz[1]) for i in range(0,3)]
        self.emb_4=nn.Embedding(4,emb_sz[2]) 
        inp_sz=3*emb_sz[0]+3*emb_sz[1]+1*emb_sz[2]+inp_sz
        self.layer1=nn.Linear(inp_sz,hdn_sz[0])
        self.layer2=nn.Linear(hdn_sz[0],hdn_sz[1])
        self.layer3=nn.Linear(hdn_sz[1],out_sz)
        self.activation=nn.Tanh()
        self.dropout=nn.Dropout(p=0.5)
        print("Network intialised inp_sz={},hdn_sz={},Emb_lays=emb2={},emb3={}".format(inp_sz,hdn_sz,len(self.emb_2),
                                                                                              len(self.emb_3),))
    def forward(self,con_x,cat_2,cat_3,cat_4):
        catl_2=[]
        catl_3=[]
        catl_4=[]
        for i in range(cat_2.shape[0]):
#            print(cat_3[i])s
            catl_2.append(torch.cat([self.emb_2[0](cat_2[i][0]),\
                                     self.emb_2[1](cat_2[i][1]),\
                                     self.emb_2[2](cat_2[i][2])]))
            #catl_2.append(self.emb_2[1](cat_2[i][1]))
            #catl_2.append(self.emb_2[2](cat_2[i][2]))
            #print(cat,cat[0],cat[1],cat[2])
            #print(self.emb_3,self.emb_3[0](cat[0]),self.emb_3[1](cat[1]),self.emb_3[2])
            catl_3.append(torch.cat([self.emb_3[0](cat_3[i][0]),\
                                     self.emb_3[1](cat_3[i][1]),\
                                     self.emb_3[2](cat_3[i][2])]))
            #catl_3.append(self.emb_3[1](cat_3[i][1]))
            #catl_3.append(self.emb_3[2](cat_3[i][2]))
            catl_4.append(self.emb_4(cat_4[i][0]))
        #print(catl_2)
#        print(len(catl_2),len(catl_3),len(catl_4))
        #print(torch.stack(catl_2).shape)
        #print(catl_2)
        #print(torch.stack(catl_2,dim=0).shape,torch.stack(catl_3).shape,torch.stack(catl_4).shape)
        #print(torch.cat([torch.stack(catl_2,dim=0),torch.stack(catl_3),torch.stack(catl_4)],dim=1).shape)
        cat_x=torch.cat([torch.stack(catl_2,dim=0),torch.stack(catl_3),torch.stack(catl_4)],dim=1)
        #cat_x=torch.cat([cat_2,cat_3,cat_4])
        #print(cat_x)
        #print(cat_x.shape,con_x.shape)
        #print(torch.cat([cat_x,con_x],dim=1).shape)
        x=torch.cat([cat_x,con_x],dim=1)
        #x=torch.cat([cat_x,con_x])
#        print(x.shape)
        x=self.activation(self.layer1(x))
        x=self.dropout(x)
        x=self.activation(self.layer2(x))
        x=self.dropout(x)
        x=F.sigmoid(self.layer3(x))
        return x    
    
class HealthDataset(dataset.Dataset):
    def __init__(self,df,y):
        df=crt_emb(df)
        self.uniq2=df[['sex', 'fasting_blood_sugar', 'exercise_induced_angina']].values
        self.uniq3= df[['resting_electrocardiographic_results', 'slope', 'thal']].values
        self.uniq4= df[['number_of_major_vessels']].values
        self.con=df[['age','chest','resting_blood_pressure','serum_cholestoral','maximum_heart_rate_achieved','oldpeak']].values
        self.y=y.values
    def __getitem__(self,ind):
        return torch.FloatTensor(self.con[ind,:]),\
                            self.uniq2[ind,:],\
                            self.uniq3[ind,:],\
                            self.uniq4[ind,:],\
                            self.y[ind]
    
    def __len__(self):
        return self.uniq2.shape[0]

def train_loop(num_epochs,model,trn_loader,val_loader,optimizer,loss_fn,writer):
    mb= tqdm(range(num_epochs),desc='epoch')

    loss_list=[]

    for t in mb :#epochs
        for idx,(x,x2,x3,x4,y_act) in enumerate(tqdm(trn_loader,desc='learning data')):
            optimizer.zero_grad()
            #print(x,x2,x3,x4)
            y_pred = model(x,x2,x3,x4)
#            print(y_pred.shape,y_act.shape)
            loss = loss_fn(y_pred, y_act)
            loss_list.append(loss.item())
#            print('loss',loss.item())
            writer.add_scalar('trainloss',loss.item(),idx+len(trn_loader)*t)
#            print('models grad',[p.grad for p in Reg.parameters()])
            if y_pred.grad:

               

                writer.add_histogram('pred_gradients',y_pred.grad.numpy())

            # Zero gradients, perform a backward pass, and update the weights.



            loss.backward()

            optimizer.step()

#            writer.add_histogram('actual',y_act.detach().numpy())

#            writer.add_histogram('pred',y_pred.detach().numpy())
#
        for i2 in range(0,len(model.emb_2)):
            writer.add_embedding(list(model.emb_2[i2].parameters())[0],global_step=t*100+idx+i2)
        for i3 in range(0,len(model.emb_3)):
            writer.add_embedding(list(model.emb_3[i3].parameters())[0],
                                 global_step=t*1000+idx+i2+i3+1)
        writer.add_embedding(list(model.emb_4.parameters())[0],global_step=t*10000+idx+i2+i3+1)
#        writer.add_embedding(list(model.emb_layers.parameters())[0],global_step=t+idx)
            
        val_loop(model,t,val_loader,loss_fn,writer)

#        plt.plot(loss_list)
#    
#        plt.title('train-loss')
#    
#    #    plt.savefig('train_loss.png')
#    
#        plt.show()

    return model

 

def val_loop(model,t,val_loader,loss_fn,writer):
    model.eval()
    val_loss=[]
    for idx,(x,x2,x3,x4,y_act) in enumerate(tqdm(val_loader,desc='validation data')):

        y_pred=model(x,x2,x3,x4)
#        act.extend(y_act.detach().numpy().tolist())
#        pred.extend(y_pred.detach().numpy().tolist())
        loss = loss_fn(y_pred, y_act)
        
#       
        writer.add_scalar('valloss',loss.item(),idx+len(val_loader)*t)

        val_loss.append(loss.item())

#    plt.plot(val_loss)
#    plt.title('val-loss')
#    plt.show()

    return val_loss
def run(train_batch_size, val_batch_size, epochs, lr,log_interval, log_dir):
    train_loader, val_loader = get_loaders(train_batch_size, val_batch_size)
    model=HeartDisDet(hdn_sz=[50,100,10])
    writer = create_summary_writer(model, train_loader, log_dir)
#    device = 'cpu'

#    if torch.cuda.is_available():
#        device = 'cuda'

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn= nn.CrossEntropyLoss()
    train_loop(epochs,model,train_loader,val_loader,optimizer,loss_fn,writer)
    torch.save(model,open(r'cat_embed.pkl','wb'))
    writer.close()


if __name__=='__main__':
    df=pd.read_csv(r'train.csv')
    df=df[:1000]
    X_train, X_test, y_train, y_test =train_test_split(df.iloc[:,:-1],df['class'])
#    X_train=X_train.reset_index()
#    X_test=X_test.reset_index()
#    y_train=y_train.reset_index()
#    y_test=y_test.reset_index()
    
#    print(X_train.shape,X_test.shape)
    run(500,500,10,0.01,1000,'logs\\')
    
#    model=HeartDisDet(hdn_sz=[50,100,10])
#    trnl,vall=get_loaders(10,10)
#    NeuralNetClassifier(model,i)