import numpy as np
import time

class Util(object):
    '''
        Purpose: Load the dataset in an object.
        Usage: dataset=Util.load_dataset(filename)
    '''
    @staticmethod
    def load_dataset(filename):
        return np.loadtxt(filename, delimiter=',', dtype=np.uint32)
    '''
        Purpose: Take a dataset having zeros and ones and return all pairwise counts 
        For each variable x and y, we return 
            xy_counts[0,0]; xy_counts[0,1]; xy_counts[1,0]; and xy_counts[1,1]
        Alert: No Laplace correction is performed.
    '''
    @staticmethod
    def compute_xycounts(dataset):
        nvariables=dataset.shape[1]
        prob_xy = np.zeros((nvariables, nvariables, 2, 2))
        prob_xy[:, :, 0, 0] = np.einsum('ij,ik->jk', (dataset == 0).astype(int), (dataset == 0).astype(int))
        prob_xy[:, :, 0, 1] = np.einsum('ij,ik->jk', (dataset == 0).astype(int), (dataset == 1).astype(int))
        prob_xy[:, :, 1, 0] = np.einsum('ij,ik->jk', (dataset == 1).astype(int), (dataset == 0).astype(int))
        prob_xy[:, :, 1, 1] = np.einsum('ij,ik->jk', (dataset == 1).astype(int), (dataset == 1).astype(int))
        return prob_xy

    '''
        Purpose: Take a dataset having zeros and ones and return marginal counts 
        For each variable x, we return x_count[0] and x_count[1]
        Alert: No Laplace correction is performed.
    '''
    @staticmethod
    def compute_xcounts(dataset):
        nvariables = dataset.shape[1]
        prob_x = np.zeros((nvariables, 2))
        prob_x[:,0]=np.einsum('ij->j',(dataset == 0).astype(int))
        prob_x[:,1] = np.einsum('ij->j',(dataset == 1).astype(int))
        return prob_x

    '''
        Purpose: Take a weighted dataset having zeros and ones and return all pairwise counts
        For each variable x and y, we return weighted
            xy_counts[0,0]; xy_counts[0,1]; xy_counts[1,0]; and xy_counts[1,1]
        Alert: No Laplace correction is performed.
    '''
    @staticmethod
    def compute_weighted_xycounts(dataset,weights):
        nvariables=dataset.shape[1]
        prob_xy = np.zeros((nvariables, nvariables, 2, 2))
        prob_xy[:, :, 0, 0] = np.einsum('ij,ik->jk', (dataset == 0).astype(int)* weights[:, np.newaxis], (dataset == 0).astype(int))
        prob_xy[:, :, 0, 1] = np.einsum('ij,ik->jk', (dataset == 0).astype(int)* weights[:, np.newaxis], (dataset == 1).astype(int))
        prob_xy[:, :, 1, 0] = np.einsum('ij,ik->jk', (dataset == 1).astype(int)* weights[:, np.newaxis], (dataset == 0).astype(int))
        prob_xy[:, :, 1, 1] = np.einsum('ij,ik->jk', (dataset == 1).astype(int)* weights[:, np.newaxis], (dataset == 1).astype(int))
        return prob_xy

    '''
        Purpose: Take a weighted dataset having zeros and ones and return marginal counts 
        For each variable x, we return weighted x_count[0] and x_count[1]
        Alert: No Laplace correction is performed.
    '''
    @staticmethod
    def compute_weighted_xcounts(dataset,weights):
        nvariables = dataset.shape[1]
        prob_x = np.zeros((nvariables, 2))
        prob_x[:,0]=np.einsum('ij->j',(dataset == 0).astype(int) * weights[:, np.newaxis])
        prob_x[:,1] = np.einsum('ij->j',(dataset == 1).astype(int) * weights[:, np.newaxis])
        return prob_x
    '''
        Purpose: Normalize a Bivariate Distribution
    '''
    @staticmethod
    def normalize2d(xycounts):
        xycountsf=xycounts.astype(np.float64)
        norm_const=np.einsum('ijkl->ij',xycountsf)
        return xycountsf/norm_const[:,:,np.newaxis,np.newaxis]

    '''
        Purpose: Normalize a Univariate Distribution
    '''
    @staticmethod
    def normalize1d(xcounts):
        xcountsf = xcounts.astype(np.float64)
        norm_const = np.einsum('ij->i', xcountsf)
        return xcountsf/norm_const[:,np.newaxis]

    '''
        Purpose: Normalize the weights
    '''
    @staticmethod
    def normalize(weights):
        norm_const=np.sum(weights)
        return weights/norm_const

    '''
        Purpose: Compute the edge weights for the Chow-Liu Algorithm
    '''
    @staticmethod
    def compute_MI_prob(p_xy,p_x):
        p_x_r = np.reciprocal(p_x)
        sum_xy=np.zeros((p_x_r.shape[0], p_x_r.shape[0]))
        sum_xy += p_xy[:,:,0,0]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,0,0],p_x_r[:,0],p_x_r[:,0]))
        sum_xy += p_xy[:,:,0,1]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,0,1],p_x_r[:,0],p_x_r[:,1]))
        sum_xy += p_xy[:,:,1,0]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,1,0],p_x_r[:,1],p_x_r[:,0]))
        sum_xy += p_xy[:,:,1,1]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,1,1],p_x_r[:,1],p_x_r[:,1]))
        return sum_xy
    

