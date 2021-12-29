#!/usr/bin/env python
# coding: utf-8

# In[13]:



"""
Define the Chow_liu Tree class
"""

#
import os
#from __future__ import print_function
import numpy as np
from Util import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order
import sys
import time
import pandas as pd


'''
Class Chow-Liu Tree.
Members:
    nvariables: Number of variables
    xycounts: 
        Sufficient statistics: counts of value assignments to all pairs of variables
        Four dimensional array: first two dimensions are variable indexes
        last two dimensions are value indexes 00,01,10,11
    xcounts:
        Sufficient statistics: counts of value assignments to each variable
        First dimension is variable, second dimension is value index [0][1]
    xyprob:
        xycounts converted to probabilities by normalizing them
    xprob:
        xcounts converted to probabilities by normalizing them
    topo_order:
        Topological ordering over the variables
    parents:
        Parent of each node. Parent[i] gives index of parent of variable indexed by i
        If Parent[i]=-9999 then i is the root node
'''
class random_forest():
    def __init__(self):
        self.nvariables = 0
        self.xycounts = np.ones((1, 1, 2, 2), dtype=int)
        self.xcounts = np.ones((1, 2), dtype=int)
        self.xyprob = np.zeros((1, 1, 2, 2))
        self.xprob = np.zeros((1, 2))
        self.topo_order = []
        self.parents = []

    '''
        Learn the structure of the Chow-Liu Tree using the given dataset
    '''
    def learn(self, dataset,k,r):
        self.nvariables = dataset.shape[1]
        self.xycounts = Util.compute_xycounts(dataset) + 1 # laplace correction
        self.xcounts = Util.compute_xcounts(dataset) + 2 # laplace correction
        self.xyprob = Util.normalize2d(self.xycounts)
        self.xprob = Util.normalize1d(self.xcounts)
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util.compute_MI_prob(self.xyprob, self.xprob) * (-1.0)
        edgemat[edgemat == 0.0] = 1e-20  # to avoid the case where the tree is not connected
        random_indices1 = np.random.choice(edgemat.shape[0], size=r, replace=False)
        random_indices2 = np.random.choice(edgemat.shape[1], size=r, replace=False)
        edgemat[random_indices1,random_indices1]=0
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert the spanning tree to a Bayesian network
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)  


    def computeLL(self, datase,k,r,rrlist):
        p_d=0.0
        for i in range(dataset.shape[0]):
            sm=0.0
            for j in range(k):
                sm = sm + 1/k * rrlist[j].getProb(dataset[i])
            p_d = p_d + np.log(sm)
        return p_d/dataset.shape[0]

    def getProb(self,sample):
        prob = 1.0
        for x in self.topo_order:
            assignx = sample[x]
            # if root sample from marginal
            if self.parents[x] == -9999:
                prob *= self.xprob[x][assignx]
            else:
                # sample from p(x|y)
                y = self.parents[x]
                assigny = sample[y]
                prob *= self.xyprob[x, y, assignx, assigny] / self.xprob[y, assigny]
        return prob


'''
    You can read the dataset using
    dataset=Util.load_dataset(path-of-the-file)
   
    To learn Chow-Liu trees, you can use
    clt=CLT()
    clt.learn(dataset)
    
    To compute average log likelihood of a dataset, you can use
    clt.computeLL(dataset)/dataset.shape[0]
'''


# In[ ]:


file=sys.argv[1]
maxll=-100000
rrlist =[]
dataset=Util.load_dataset('/Users/mohamedabrar/Downloads/dataset/'+file+'.ts.data')
valset=Util.load_dataset('/Users/mohamedabrar/Downloads/dataset/'+file+'.valid.data')
testset=Util.load_dataset('/Users/mohamedabrar/Downloads/dataset/'+file+'.test.data')
df = pd.DataFrame(columns = ['Dataset','k','r','LogLikelihood'])
i=0
for k in [10,50,100,200]:
    #for r in [10,20,50,100]:   
    #for r in [10,20,30,40]:    #for plants, kdd
    for r in [5,10,12,15]:    #for nltcs, msnbc
        #take care of value of r since some datasets have less no. of features 
        rrlist =[]
        for i in range(k):
            number_of_rows = dataset.shape[0]
            random_indices = np.random.choice(number_of_rows, size=number_of_rows, replace=True)
            sample = dataset[random_indices, :]
            rr=random_forest()
            rr.learn(sample,k,r)
            rrlist.append(rr)
        ll=rr.computeLL(valset,k,r,rrlist)
        if ll>maxll:
            maxll=ll
            maxr=r
            maxk=k
        df.loc[i] = [file,k,r,ll]
        print([file,k,r,ll])
        i=i+1
print('Best Parameter k= ',maxk,' r= ',maxr) 
rrlist =[]
data=np.concatenate((dataset, valset))
for i in range(maxk):
    number_of_rows = data.shape[0]
    random_indices = np.random.choice(number_of_rows, size=number_of_rows, replace=True)
    sample = data[random_indices, :]
    rr=random_forest()
    rr.learn(sample,maxk,maxr)
    rrlist.append(rr)
ll=rr.computeLL(testset,maxk,maxr,rrlist)

print(df)
print('Best Param likelihood=',ll,'for k= ',maxk,'for r= ',maxr)

# In[ ]:




