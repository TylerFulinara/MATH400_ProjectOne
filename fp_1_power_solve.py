#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 21:13:22 2023

@author: hboateng
"""
from numpy import ones, nonzero
from numpy import linalg as LA

def sortPageRank(prdict):
    return sorted(prdict.items(), key = lambda item: item[1], \
                  reverse=True)
    
class pageRank():
    def __init__(self,G,nodelist,eps,tol,maxiter):
        
        n = G.shape[0]
        for i in range(n): # normalize columns of A
            ci = LA.norm(G[:,i],1)
            if ci > 0:
                G[:,i]=G[:,i]/ci
            else: # adjustment for a column of zeros
                G[:,i]=ones((n,))/float(n)           
            
        self.G       = G         # normalized matrix        self.nodes   = nodelist  # list of node labels
        self.eps     = eps       # probability of jumping to a link on page
        self.size    = G.shape[0]# size of matrix
        self.tol     = tol # tolerance for power method
        self.maxiter = maxiter # maximum number of iterations for power method
        if not nodelist: # list of node labels
            self.nodes = [k for k in range(self.size)]
        else:
            self.nodes = nodelist
    
    def powermethod(self):
        n = self.size
        
        # Get sparse G (as a list)
        
        #list of lists of index of nonzero elements in each row
        nzre = [nonzero(self.G[k,:]>0) for k in range(n)] 
        
        #list of vectors of nonzero elements in each row
        nzv = [self.eps*self.G[k,nzre[k]] for k in range(n)]
        
        #for k in range(n):
         #   print(f"nzre = {nzre[k]}, nzv = {nzv[k]}")
        
        oeps = (1.0-self.eps)/n
        x = ones((n,1))/float(n) # initial vector

        
        xn1 = LA.norm(x,1)
        ntol = xn1
        niter = 0
        while ntol > self.tol and niter < self.maxiter :
            xold = x       
            for k in range(n):
                x[k] = nzv[k]@x[nzre[k]] + oeps
                
            xn1  = LA.norm(x,1)
            x    = x/xn1
            ntol = LA.norm(x-xold,1)
            
            niter += 1
            print(f"n = {niter}, ntol = {ntol}, x = {x}")
            
        return sortPageRank({k:float(v) for (k,v) in zip(self.nodes,x)})