# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:19:15 2018

@author: Arunodhaya
"""

import torch
import pandas as pd
from torch.utils.data import dataset
def crt_emb(df):
    df['thal']=df['thal'].replace({3:0,6:1,7:2})
    df['slope']=df['slope'].replace({1:0,2:1,3:2})
    df['exercise_induced_angina']=df['exercise_induced_angina'].replace({1:0,2:1,3:2})
    return df

class HealthDatasetPred(dataset.Dataset):
    def __init__(self,df):
        df=crt_emb(df)
        self.uniq2=df[['sex', 'fasting_blood_sugar', 'exercise_induced_angina']].values
        self.uniq3= df[['resting_electrocardiographic_results', 'slope', 'thal']].values
        self.uniq4= df[['number_of_major_vessels']].values
        self.con=df[['age','chest','resting_blood_pressure','serum_cholestoral','maximum_heart_rate_achieved','oldpeak']].values
#        self.y=y.values
    def __getitem__(self,ind):
        return torch.FloatTensor(self.con[ind,:]),\
                            self.uniq2[ind,:],\
                            self.uniq3[ind,:],\
                            self.uniq4[ind,:]
    
    def __len__(self):
        return self.uniq2.shape[0]

testdf=pd.read_csv(r'test.csv')
testdf.shape
test=HealthDatasetPred(testdf)
test_ldr=dataloader.DataLoader(test,batch_size=testdf.shape[0])
tst_ldr=iter(test_ldr)
con,x2,x3,x4=next(tst_ldr)
model=torch.load('cat_embed.pkl')

pred=model(con,x2,x3,x4)
from torch.distributions import categorical
cat=categorical.Categorical(pred)
res=cat.sample()

testdf.index
testdf['class']=res
testdf['class'].to_csv('cat_embed.csv')













