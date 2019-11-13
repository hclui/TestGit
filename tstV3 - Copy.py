# tst3.0
# contains subroutines for testing
import math
import numpy as np
import copy
from scipy.spatial import distance

from ConstantsV3 import NTRIAL, DIST2BOUND, DIST_INTO_CELL, Knn


def predict(Xtoken, Rules, nF):
    X = copy.deepcopy(Xtoken)
    # if X[i] is out-of-bound, move it back ...
#    for i in range(0,nF):
#        if X[i]<0.0 :
#            X[i]=DIST2BOUND  # make sure it is inside zone
#        if X[i]>1.0 :
#            X[i]=1.0 - DIST2BOUND # make sure it is inside zone
    
    result = nF   # one beyong the max # of class --> No rule would have this class
    for Ridx in range(0,len(Rules)):
        Fmin = Rules[Ridx].Fmin
        Fmax = Rules[Ridx].Fmax

        for j in range(0,nF):
            flag = (Fmin[j] <= X[j]) and (X[j] <= Fmax[j])
            if not flag:
                break

        if (j == nF-1) and (flag) : # if X satisfies all inequalities of Rule[Ridx]....
            result = Rules[Ridx].cls # class label of this rule
            break

    if result == nF: # this appears never happen????
        result = kNN(Xtoken, Rules[Ridx].data,Rules[Ridx].target,Knn, nC) 
    return result, Ridx

def fNNVR(X, Ridx, Rules, nF):
# fNNVR (find-nearest-non-void-rule) is entered if a rule having a void class is encountered
# it will compute the distances of sample X to all boundaries of this void rule R
# sort the distances and put to stack (DPstack)
# pop the stack to get the min. distance entry
# find the rule sharing the same boundary
# if the new rule is not a void class, return the class and new-rule index
# else go back and pop the stack to examine the next entry

# this version won't explor the adjacent rules of the new rules if the new rule is a void-class rule
    R = Rules[Ridx]
    DP = []
    # NOTE: should work if checking active features in Fset only, no need for all feature set
    for f in R.Fset:  # for all features in Fset
        if f == -1:
            continue  # skip 
        #if (X[f] > DIST2BOUND) :  # ignore pts at global boundaries
        if (X[f] > DIST2BOUND) and (R.Fmin[f] > 0.0):  # ignore pts at global boundaries
            D = (X[f] - R.Fmin[f])**2  # distance
            DP.append((D,f,0))  # '0' indicates from Fmin
        #if (X[f] < 1.0 - DIST2BOUND) : # ignore pts at global boundaries
        if (X[f] < 1.0 - DIST2BOUND) and (R.Fmax[f] < 1.0): # ignore pts at global boundaries
            D = (X[f] - R.Fmax[f])**2
            DP.append((D,f,1))  # '1' indicates from Fmax
        
    DPstack = sorted(DP)    # sort DP using the 1st element of tuple -- distance

    while len(DPstack) > 0:
        (D, f, k) = DPstack.pop(0)
#        if (R.Fmin[f] == 0) or (R.Fmax[f] == 1): # skip if feature at boundary
#            continue
        
        R2chk = []  # indices of new rules
        for r in range(0,len(Rules)):
#            Fmin = Rules[r].Fmin
#            Fmax = Rules[r].Fmax
            if k == 0:  # only need to check Fmax
                if Rules[r].Fmax[f] == R.Fmin[f]: # upper boundary of f same as lower boundary of R
                    R2chk.append(r)
            else:       # only need to check Fmin
                if Rules[r].Fmin[f] == R.Fmax[f]: # lower boundary of f same as upper boundary of R
                    R2chk.append(r)
        
        nRidx = []
        for i in range(0, len(R2chk)):
            flag = True
            ii = R2chk[i]
            for j in range(0,nF):
#                if (j == k): # no need to check current feature
                if (j == f): # no need to check current feature
                    continue
                if (Rules[ii].Fmin[j] > X[j]) or (Rules[ii].Fmax[j] < X[j]): # violating other feature boundaries?
                    flag = False
                    break
#                else:
#                    nRidx.append(ii)   # found new rule THERE SHOULD BE ONLY ONE OF THEM
                    
#                
            if flag: # satisfying all other boundary constraints
                nRidx.append(ii)   # found new rule THERE SHOULD BE ONLY ONE OF THEM
#        
        if len(nRidx) != 1:
            print("More than one New rule found")
        
        r = nRidx[0]
        if Rules[r].cls != -1: # NOT a void class
            return Rules[r].cls, r
        else:
            print("next..")
        
    if len(DPstack) == 0: # ALL rules in DPstack are void class !!!
        return -1, -2
    
def resolve(X, Ridx, Rules, nF):
    Fmin = Rules[Ridx].Fmin
    Fmax = Rules[Ridx].Fmax
    Fminmax = np.concatenate((Fmin, Fmax))
    Fminmax = Fminmax.reshape(2,len(Fmin))
    nX, dx, fidx, mdi = fndNC(X, Fminmax, nF) # move X to nX by finding closest rectangular edge
    result, nRidx = predict(nX, Rules, nF)
    print("@@@@resolve: Ridx:%d ,nRidx:%d, X:%f,%f; nX:%f,%f"%(Ridx,nRidx, X[0], X[1],nX[0],nX[1]))

    trial = 0
    while (result < 0) and (trial < NTRIAL):
        Fminmax = copy.deepcopy(Rules[Ridx][2])     # original rectangular cube
        nFminmax = Rules[nRidx][2]
        if mdi%2 != 0: # odd
            # try to avoid looping
            tmp = nFminmax[fidx,1]-X[fidx] # use the far end edge to update dx on same factor
            if tmp != dx[mdi]:
                dx[mdi]=tmp
            else:
                dx[mdi]= 1.0    # make it a large value not to be selected
            Fminmax[fidx,1] = nFminmax[fidx,1] # change the original boundary of this edge
        else: # even
            tmp = X[fidx]-nFminmax[fidx,0]
            if tmp != dx[mdi]:
                dx[mdi]=tmp
            else:
                dx[mdi]= 1.0    # make it a large value not to be selected
            Fminmax[fidx,0] = nFminmax[fidx,0]
        
        mdi = np.argmin(dx)
        fidx = mdi//2   # integer division
        nX = copy.deepcopy(X)
        if mdi%2 != 0: # odd
            nX[fidx] = Fminmax[fidx,1]+DIST_INTO_CELL
        else: # even
            nX[fidx] = Fminmax[fidx,0]-DIST_INTO_CELL
        result, nRidx = predict(nX, Rules, nF) # do prediction again till it works.
        print("@@@@resolve: Ridx:%d ,nRidx:%d, X:%f,%f; nX:%f,%f"%(Ridx,nRidx, X[0], X[1],nX[0],nX[1]))
        trial += 1

    if result < 0:  # keep falling to void cells...
        FatherNode = Rules[Ridx].father # get father node
        D = FatherNode.dataset
        T = FatherNode.target
        result = kNN(X, D,T, Knn, nC)
        nRidx = -1
    
    return result, nRidx

def fndNC(X, Fminmax, nF): # find nearest cell
#### !!!! this routine needs to be modified for multi-dim feature data
# current version can select a feature that has never been used.
# may need to restrict it to Fset features only.
# but it is a krudge
    h = 1e-5
    dx = np.zeros(2*nF)
    for i in range(nF):
        low = Fminmax[i][0]
        high = Fminmax[i][1]
        dx[2*i] = X[i]-low
        dx[2*i+1] = high - X[i]
        
    mdi = np.argmin(dx)
    fidx = mdi//2   # integer division
    nX = copy.deepcopy(X)
    if mdi%2 != 0: # odd
        nX[fidx] = Fminmax[fidx,1]+DIST_INTO_CELL
    else: # even
        nX[fidx] = Fminmax[fidx,0]-DIST_INTO_CELL
        
    return nX, dx, fidx, mdi


def kNN(X, D, T, K, nC): 
# X:   current data point
# D:   set of (training) data
# T:   targets of training data
# K:   nearest K neighborhood

    dist = np.zeros(len(D))
    for i in range(len(D)):
        if all(D[i,:-1] == X[:-1]):    # test token same?
            dist[i]= 1.e6 # make it a large number so it won't be selected
        else:
            dist[i]= distance.euclidean(X,D[i])
            
    sDist = np.sort(dist)
    sDistIdx = np.argsort(dist)
    
    TopKIdx = sDistIdx[0:K-1]  # top K entries
    
    clscnt = np.zeros(nC)
    for i in range(0,nC):
        idx = y[TopKIdx[i]]
        clscnt[idx] += 1
        
    winner = np.argmax(clscnt)
    return winner
    
#    if len(dist)>=3:    
#        min3 = np.argpartition(dist,[0,1,2])[0:3] # first 3 min dist values
#    # need to modify the following statements to extend for n-class case
#        Z = np.sum(T[min3])
#        if Z <=0 :
#            result = 0
#        else:
#            result = 1
#    else: # less than 3 tokens ..
#        minidx =  np.argmin(dist)
#        result = T[minidx]+0.5
#    print("  !!! kNN: len of targets: %d, result= %d" % (len(T), result))   
    return result

# need to define a class for Rules (i.e. Rules)
# need to separate Fminmax to Fmin & Fmax or LowerBound & UpperBound




#def fNNVR(X, cRidx, Rules):
#    # find nearest boundary
#    Fminmax = Rules[cRidx][2]   # get Fminmax of cRidx
#    
#    for f in active_feature_set:
#        d2b[f][0] = X - Fminmax[f][0]
#        d2b[f][1] = Fminmax[f][1] - X
#        
#    NearestBoundary = np.sort(d2b)
#    NBIdx = np.argsort(d2b)
#    # form tuples of (feature[i], Bvalue)
#    
#    for each tuple element:
#        # search rules whose boundary the same as this one (for same feature)
#        for r in range(0,len(Rules)):
#            if Rules[i].Fminmax[f][0] == Bvalue:
#                Rule2Chk.append(i)
#            if Rules[i].Fminmax[f][0] == Bvalue:
#                Rule2Chk.append(i)
#        
#        nChk = len(Rule2Chk)
#        for j in range(0,nChk):
#            # find one that satisfy other Fminmax requrements
#            # for each active features except one in consideration
#            # check if X satisfies other feature boundaries
#        
#        # only one rule survives -- call Rstar
#        
#        If Rules[Rstar].class ~= Void
#            return Rstar
#        else:
#            # for this rule
#            # for high/low boundary of this feature
#            # find corner point and distance of X to this corner point
#            #select the min and register the rule index of this min distance rule 
#            # put to Dist2Pt array
#            
            
            