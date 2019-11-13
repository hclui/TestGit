# Factor space classifier
# version 3.0 is based on 2.2.1; see comments on 2.2.1 for details.
# this version chops the code into multiple python files.

#
import math
import numpy as np
import copy
from scipy.spatial import distance

from tstV3 import *
from Tree2RuleV3 import *
from BuildTreeV3 import BuildTree, getstate, HistSelect
from Tree2RuleV3 import Tree2Rules, RefineRules
from tstV3 import predict, resolve, fNNVR
from FS_showV3 import *
from nodeV3 import node
from ruleV3 import rule
from ConstantsV3 import DIST2BOUND
class FactorSpaceClassifierV3:
    
    def __init__(self):
        pass
    
    def fit(self,X,y):
        nC = len(np.unique(y))   # no of classes
        nF = X.shape[1]     # no of factors (2nd dim of X array)
        
        self.sf = np.zeros((nF,2), dtype=float)
        for i in range(0,nF):
            Xmin = X[:,i].min()
            Xmax = X[:,i].max()
            self.sf[i,0] = 1./(Xmax-Xmin)
            self.sf[i,1] = Xmin

# normalize data to [0,1] interval        
        nX = copy.deepcopy(X)
        for i in range(0,len(X)):
            for j in range(0,nF):
                nX[i,j] = (nX[i,j]-self.sf[j,1])*self.sf[j,0]
        
        Fminmax = np.zeros((nF,2)) # holds the min & max value of each facotr
        Fmin = np.zeros(nF)
        Fmax = np.zeros(nF)
        h = 0  # used only in adding a pad but set to zero after doing normaliztion !! see rangedata()
        for i in range(0,nF):
            Fminmax[i,0] = nX[:,i].min() - h
            Fmin[i] = nX[:,i].min() - h
            Fminmax[i,1] = nX[:,i].max() + h
            Fmax[i] = nX[:,i].max() + h
        oFidx = -1
        Fidx, partitions = HistSelect(nX, y, Fminmax, Fmin, Fmax, nF, nC, oFidx, 0, 1)
        state = getstate(nX, y, nC)
        Fseq = [Fidx]
        Fset = np.unique(Fseq)
        TreeRoot = node(Fidx, Fseq, Fset, partitions, Fminmax, Fmin, Fmax, state, nX, y)
        BuildTree(TreeRoot, nF, nC)
        #PrintTree(TreeRoot, nF)
#        self.RuleInfo = Tree2Rules(TreeRoot, nF, nC)
#        RefineRules(self.RuleInfo, nF)
        self.Rules = Tree2Rules(TreeRoot, nF, nC)        
        RefineRules(self.Rules, nF)
        if nF == 2:     # show decision regions only for 2 factor situation
            ShowRectangles(nX, y, self.Rules, 0, 1, nF)
#       CreateRules(self.RuleInfo, nF)
        score = 0
        for i in range(len(nX)):
            result, Ridx = predict(nX[i,:], self.Rules, nF)
#            print("X[%d]: [%f,%f] predict: %d, actual: %d" % (i, X[i,0],X[i,1], result, y[i]))

            if (result < 0):
                print(">>>before resolve: X[%d]: [%f,%f] predict: %d, actual: %d by rule %d" %
                      (i, nX[i,0], nX[i,1], result, y[i], Ridx))
                #print("   X[%d]: [%f,%f] predict: %d, actual: %d" % (i, nX[i,0],nX[i,1], result, y[i]))
                Oresult, ORidx = resolve(nX[i,:], Ridx, self.Rules, nF)
                result, Ridx = fNNVR(nX[i,:], Ridx, self.Rules, nF)
                print("<<<after  resolve: result = %d by rule %d" % (result, Ridx))

            if result == y[i]:
                score += 1
        accuracy = score / len(y)
        print("training accuracy:", accuracy)

        return #self
    
    def score(self,X,y):
        print("\n\nTest scoring")
        score = 0
        nF = X.shape[1]     # no of factors (2nd dim of X array)
# normalize data to [0,1] interval        
        nX = copy.deepcopy(X)
        for i in range(0,len(X)):
            for j in range(0,nF):
                nX[i,j] = (nX[i,j]-self.sf[j,1])*self.sf[j,0]
                if nX[i,j]<0.0 :
                    nX[i,j]=DIST2BOUND  # make sure it is inside zone
                if nX[i,j]>1.0 :
                    nX[i,j]=1.0 - DIST2BOUND # make sure it is inside zone
        
        for i in range(len(nX)):
            result, Ridx = predict(nX[i,:], self.Rules, nF)
#            print("X[%d]: [%f,%f] predict: %d, actual: %d" % (i, X[i,0],X[i,1], result, y[i]))

            if (result < 0):
                print(">>>before resolve: X[%d]: [%f,%f] predict: %d, actual: %d by rule %d" %
                      (i, nX[i,0], nX[i,1], result, y[i], Ridx))
                #print("   X[%d]: [%f,%f] predict: %d, actual: %d" % (i, nX[i,0],nX[i,1], result, y[i]))
                #result, Ridx = resolve(nX[i,:], Ridx, self.Rules, nF)
                result, Ridx = fNNVR(nX[i,:], Ridx, self.Rules, nF)
                print("<<<after  resolve: result = %d by rule %d" % (result, Ridx))

            if result == y[i]:
                score += 1
            else: # print error tokens
                print("ERR::X[%d]: [%f,%f] predict: %d, actual: %d; Rule %d" % (i, nX[i,0],nX[i,1], result, y[i], Ridx))
        
        accuracy = score / len(y)
        print("test accuracy:", accuracy)
        if nF == 2:     # show decision regions only for 2 factor situation
            ShowRectangles(nX, y, self.Rules, 0, 1, nF)
        return accuracy

