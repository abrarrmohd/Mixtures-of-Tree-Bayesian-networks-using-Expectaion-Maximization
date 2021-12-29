#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order
import sys
import time
from Util import *
from CLT_class import CLT
import random
import sys

class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks    

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components, max_iter, epsilon):
        self.n_components=n_components
        # For each component and each data point, we have a weight
        weights=np.zeros((n_components,dataset.shape[0]))
        randomlist = []
        for i in range(n_components):
            n = random.randint(1,5)
            randomlist.append(n)
        p_z = [i/sum(randomlist) for i in randomlist]
        print(p_z)
        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
        for i in range(n_components):
            # compute mutual information score for all pairs of variables
            # weights are multiplied by -1.0 because we compute the minimum spanning tree
            edgemat = np.random.rand(dataset.shape[1],dataset.shape[1])*(-1.0)
            edgemat[edgemat == 0.0] = 1e-20  # to avoid the case where the tree is not connected
            # compute the minimum spanning tree
            Tree = minimum_spanning_tree(csr_matrix(edgemat))
            # Convert the spanning tree to a Bayesian network
            order, par = depth_first_order(Tree, 0, directed=False)  
            clt=CLT()
            clt.topo_order=order
            clt.parents=par
            clt.nvariables=dataset.shape[1]
            clt.xycounts = Util.compute_xycounts(dataset) + 1
            clt.xcounts = Util.compute_xcounts(dataset) + 2
            clt.xyprob = Util.normalize2d(clt.xycounts)
            clt.xprob = Util.normalize1d(clt.xcounts)
            self.clt_list.append(clt)
        for itr in range(max_iter):
            #E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]
            #Your code for E-step here
            for i in range(dataset.shape[0]):
                for t in range(n_components):
                    w=self.clt_list[t].getProb(dataset[i][:])
                    weights[t][i]=w*p_z[t]
            #print(weights.shape)
            w1=weights.sum(axis=0)
            weights=weights/w1
            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            temp_p_z=p_z
            #Your code for M-Step here
            for t in range(n_components):
                self.clt_list[t].update(dataset,weights[t])
            p_z=weights.sum(axis=1)/dataset.shape[0]
            if np.all((abs(temp_p_z-p_z))<=epsilon):
                print(abs(temp_p_z-p_z))
                self.mixture_probs=p_z
                break
        print('No. of iterations=',itr)
        self.mixture_probs=p_z
    
    """
        Compute the log-likelihood score of the dataset
    """
    def computeLL(self, dataset):
        p_d=0.0
        for i in range(dataset.shape[0]):
            sm=0.0
            for j in range(self.n_components):
                sm = sm + self.mixture_probs[j]*self.clt_list[j].getProb(dataset[i])
            p_d = p_d + np.log(sm)
    
        
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        return p_d/dataset.shape[0]
    
    
    


# In[3]:


file=sys.argv[1]
n_c=int(sys.argv[2])
max_it=int(sys.argv[3])

dataset=Util.load_dataset('/Users/mohamedabrar/Downloads/dataset/'+file+'.ts.data')
mm=MIXTURE_CLT()
mm.learn(dataset,n_components=n_c,max_iter=max_it,epsilon=1e-5)


# In[4]:
print('For file=',file)

print('LOG LIKELIHOOD for Train set',mm.computeLL(dataset))


# In[5]:


Valset=Util.load_dataset('/Users/mohamedabrar/Downloads/dataset/'+file+'.valid.data')
print('LOG LIKELIHOOD for validation set',mm.computeLL(Valset))


# In[6]:


testset=Util.load_dataset('/Users/mohamedabrar/Downloads/dataset/'+file+'.test.data')
print('LOG LIKELIHOOD for Test set',mm.computeLL(testset))


# In[ ]:





# In[ ]:




