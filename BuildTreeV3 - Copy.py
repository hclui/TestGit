# BuildTreeV3
import math
import numpy as np
import copy
from scipy.spatial import distance

from nodeV3 import node
from ConstantsV3 import NDATA, NLEVEL, REMAIN, NBIN, CLS_WEIGHT, VOID, SINGLE, MIXED

#NDATA = 3
#NBIN = 5    # min. # of bins for histogram
#REMAIN = 0
#NLEVEL = 10
## global variables


def BuildTree(TreeRoot, nF, nC):
# use breath first search to build tree ...
    file = open("BuildTree.log", "w")
    buf = "Tree Root - %d data points; Fidx : %d"%(len(TreeRoot.target), TreeRoot.Fidx)
    file.write(buf + '\n')
    write2file(TreeRoot, file, nF)

    level = 0
    queue = []
    queue.append([TreeRoot, level])
    while (len(queue) != 0):
        tmp = queue[0]
        del queue[0]        # dequeue
        FatherNode = tmp[0]

        if FatherNode.state['status'] != 'mix':  # if it is occupied by > 1 classes, then ignore
             continue
        else:
             level = tmp[1]+1
             if level > NLEVEL:
                 continue

        buf = "\nTree level: %d" %(level)
        file.write(buf + '\n')
        FFidx = FatherNode.Fidx
        partitions = FatherNode.partitions
        # make child node
        NewNode = MakeNode(FatherNode, FFidx, 0, nF, nC)
        buf = "Child - %d data points; Fidx : %d"%(len(NewNode.target), NewNode.Fidx)
        file.write(buf + '\n')
        write2file(NewNode, file, nF)
        FatherNode.child = NewNode
        current = NewNode
        current.father = FatherNode
        queue.append([current, level])   # enqueue child node
        # make next nodes
        nA = len(partitions)
        for Aidx in range(1,nA):
            NewNode = MakeNode(FatherNode, FFidx, Aidx, nF, nC)
            buf = "~~~Next - %d data points; Fidx : %d"%(len(NewNode.target), NewNode.Fidx)
            file.write(buf + '\n')
            write2file(NewNode, file, nF)
            current.next = NewNode
            current.father = FatherNode
            current = NewNode
            queue.append([current, level])   # enqueue next node            

    file.close()
    return            


def MakeNode(FatherNode, Fidx, Aidx, nF, nC):

#    subdata, subtarget, Fminmax = rangedata(FatherNode, Fidx, Aidx)
    target  = FatherNode.target
# The following code is not necessary is the most efficient code...
# it uses the factor [Fidx] and attribute ranges that was determined
# in the father note to collect the data points faliing into range(AIDX)
#
# If the range and the datapoints falling into the ranges are stored in the
# father node, then the following code needs not be executed
#
# the partitions of histogram[Fidx] in HistSelect has the info of # of tokens in
# each class in each bin. summing up the bins belonging to the partitions gives same result
    
    dataset = FatherNode.dataset 
    df = dataset[:,Fidx]    # select the column for Fidx only
    Range = FatherNode.partitions[Aidx]
    low,high = Range[0], Range[1]
    
    Fminmax = copy.deepcopy(FatherNode.Fminmax)
    Fminmax[Fidx,0] = low    
    Fminmax[Fidx,1] = high
    Fmin = copy.deepcopy(FatherNode.Fmin)
    Fmin[Fidx] = low
    Fmax = copy.deepcopy(FatherNode.Fmax)
    Fmax[Fidx] = high
    
#    didx = np.where(np.logical_and(df >= low, df < high))  # index to dataset that satisifies conditions
    didx = np.where(np.logical_and(df >= low, df <= high)) # need to change to <= for high !!! to avoid missing boundary data points for now
# didx are indices of datapoints falling within range[Aidx]
    tmp = dataset[didx,:]  # for some reason, python generates a list of a 2 dim np array
    subdata = tmp[0]       # so needs to get rid of the list or else wrong dimensionality 
    subtarget = target[didx]
        
    state = getstate(subdata, subtarget, nC)
    if state['status'] == 'mix':  # if subdata contain > 1 class then need to work more...
# SHOULD USE CURRENT NODE FMINMAX below

        if (len(subdata) > NDATA):
#            newFidx, newpartitions = HistSelect(subdata, subtarget, FatherNode.Fminmax, nF, nC, Fidx, low, high)
            newFidx, newpartitions = HistSelect(subdata, subtarget, Fminmax, Fmin, Fmax, nF, nC, Fidx, low, high)
        else:
#            newFidx, newpartitions = DirectSelect(subdata, subtarget, FatherNode.Fminmax, nF, nC, Fidx, low, high)
            newFidx, newpartitions = DirectSelect(subdata, subtarget, Fminmax, Fmin, Fmax, nF, nC, Fidx, low, high)
    else:
        newFidx, newpartitions = -1, []
    
    newFseq = copy.deepcopy(FatherNode.Fseq)
    newFseq.append(newFidx)
    newFset = np.unique(newFseq)
    NewNode = node(newFidx, newFseq, newFset, newpartitions, Fminmax, Fmin, Fmax, state, subdata, subtarget)
    NewNode.father = FatherNode
    
    return NewNode

def getstate(dataset, target, nC):
    state = {}   # an empty dictionary
    ntok = 0
    for i in range(0,nC):
        m = len(np.argwhere(target==i))
        state[i] = m
        ntok += m

    cnt = list(state.values()) # get a list of counts for each class
    dominate = max(cnt)
    remain = sum(cnt)-dominate  # total count of non-domainant tokens
    cls = cnt.index(dominate)
    
    cnt1 = [i for i,e in enumerate(cnt) if e !=0] # find non-zero elements in cnt
    if len(cnt1) ==0:
        state['status'] = 'void'
        state['cls'] = -1
    elif (len(cnt1) == 1) or (remain == REMAIN):
       state['status'] = 'single'
       
       state['cls'] = cls
    else:
       state['status'] = 'mix'
       state['cls'] = nC
       
    return state

def HistSelect(dataset, target, Fminmax, Fmin, Fmax, nF, nC, oFidx, low, high):
# HistSelect should select a feature; then merges bins to partitions.
# then for each partition, for each class find # of tokens  in this partition.
# this info will eventually go to each child node. Can store actual data points & targets
    
    # create a 2-D list structure for histograms & bin edges...
    Hlist = [[]]*nF
    BElist = [[]]*nF
    hist = [[]]*nF
    sumhist = [[]]*nF
    DiscScore = np.zeros(nF)
    maxbinidx = np.zeros(nF, dtype = int)
    maxbin = np.zeros(nF, dtype = int)

    for i in range(0,nF):
        Hlist[i] = [[]]*nC
        BElist[i] = [[]]*nC
        hist[i] = [[]]*nC
    
    for i in range(0, nF):
#        if i == oFidx:
#            continue
        fmin = Fmin[i]
        fmax = Fmax[i]
        oFmin = Fminmax[i,0]
        oFmax = Fminmax[i,1]
        nbin = np.zeros(nC, dtype = int)
        
#        if i == oFidx:
#            Fmin, Fmax = low, high
        for j in range(0, nC):
            fdata = dataset[target==j][:,i] # extract data for class j, factor 
            if len(fdata) > NBIN:
                Hlist[i][j], BElist[i][j] = np.histogram(fdata, bins = 5, range=(oFmin, oFmax))
                Hlist[i][j], BElist[i][j] = np.histogram(fdata, bins = 5, range=(fmin, fmax))
            else:
                Hlist[i][j], BElist[i][j] = np.histogram(fdata, bins = NBIN, range=(oFmin, oFmax))
                Hlist[i][j], BElist[i][j] = np.histogram(fdata, bins = NBIN, range=(fmin, fmax))
            nbin[j] = len(Hlist[i][j])

        maxbin[i] = np.max(nbin)
        maxbinidx[i] = np.argmax(nbin)

        for j in range(0, nC):
        # create x-axis divisions for re-sampling....
            div_width = (fmax-fmin)/maxbin[i]
            div_width = (oFmax-oFmin)/maxbin[i]
            xdiv = np.arange(maxbin[i])*div_width + oFmin + div_width/2
            xdiv = np.arange(maxbin[i])*div_width + fmin + div_width/2
#            binsize = BElist[i][maxbinidx[i]][1] - BElist[i][maxbinidx[i]][0]
            binsize = BElist[i][j][1] - BElist[i][j][0]
            hist[i][j] = resampling(Hlist[i][j], maxbin[i], binsize, xdiv, oFmin)
            hist[i][j] = resampling(Hlist[i][j], maxbin[i], binsize, xdiv, fmin)


        # compute discriminating power for each bin
        sumhist[i] = sum(hist[i])
        dtmp = np.zeros(maxbin[i])
        for k in range(0, maxbin[i]):
            x = np.asarray(hist[i]) # convert list to numpy array
            jstar = np.argmax(x[:,k])
            if sumhist[i][k] != 0: # some token falls into this bin
                if sumhist[i][k] != hist[i][jstar][k]:
                    dtmp[k] = hist[i][jstar][k]/ (sumhist[i][k]-hist[i][jstar][k])
#                    DiscScore[i] += hist[i][jstar][k]/ (sumhist[i][k]-hist[i][jstar][k])
                else: # this bin occupied by one class only
                    dtmp[k] = hist[i][jstar][k]*CLS_WEIGHT
#                    DiscScore[i] += hist[i][jstar][k]*CLS_WEIGHT  # 2 is the importance factor !!

        K = 3 # median filter length; must be ODD number
        xtmp = np.zeros(maxbin[i])
        for k in range(1,maxbin[i]-1):
            xtmp[k] = median(dtmp[k-1:k+2], K)
        xtmp[0] = xtmp[1]
        xtmp[-1] = xtmp[-2]

#        DiscScore[i] = np.sum(xtmp)
        DiscScore[i] = np.sum(dtmp)
#        DiscScore[i] = DiscScore[i]/maxbin[i] # normalize by # of re-sampled) bins

# Find factor Fidx that has max. discriminating power            
# the following 'if' takes care of special case
    if np.all(DiscScore ==0): # if all diffhist the same and count is zero
        fset = {i for i in range(0,nF)} - {oFidx} # a set of (factors - oFix)
        Fidx = fset.pop()   # arbitrary taking the 1st element
    else: # choose the factor that yields max. sum values
        Fidx = np.argmax(DiscScore) # Fidx is the chosen Factor index

#    BinEdges = xdiv - div_width/2
#    BinEdges = np.append(BinEdges, xdiv[-1]+div_width/2)
    partitions = MergeBins(maxbin[Fidx], hist[Fidx], BElist[Fidx][maxbinidx[Fidx]], nF)

    #ShowHistograms(nF, nC, BinEdges, hist, diffhist, sumhist, Fidx, partitions)
    return Fidx, partitions

def resampling(H, maxbin, binsize, xdiv, Fmin):
# this routine unify histograms of different classes to same # of divisions
# maxbin is the max # of bins of all class histograms of the selected feature
# binsize is the bin-width of current histogram that needs to be re-sampled
# xdiv is the re-sampling points (for the current histogram)
    sdata = np.zeros(maxbin)
    for i in range(0,maxbin):
        pt = xdiv[i] - Fmin
        bin_idx = int(pt/binsize)
        sdata[i] = H[bin_idx]

    return sdata

def median(x, K):
    mxidx = np.argsort(x)
    return x[mxidx[K//2]] 
 

def MergeBins(nbin, hist, BinEdges, nF):
#    see node status assignment from ConstantsV3.py
#    void = 0    # no bin are occupied by any token
#    single = 1  # only one bin are occupied
#    mix = 2     # more than one bin are occupied
# sbin stores status of each bin:
    sbin = np.zeros(nbin, dtype = 'int') # initialize to all VOID classes
    hist = np.asarray(hist)
    for i in range(0,nbin):
        sbin[i] = np.count_nonzero(hist[:,i])
    sbin[sbin>2] = MIXED    # status == 2
# binmax stores the class index that has highest # of token falling to this bin
    binmax = np.argmax(hist, axis = 0)
    allmix = np.all(sbin >= MIXED)  # special case for further checking if all bins are occupied by > 1 classes

## NOT SURE IF THE FOLLOWING CODE IS GOOD. So comment out first
## It appears to improve accuracy quite a bit for cancer data
## but it will increase the # of paritions and hence # of horizontal nodes     
# if the bin is dominated by one class, treat it as single.
    for i in range(0,nbin):
        if sbin[i] == 2:
            dominate = binmax[i]
            remain = np.sum(hist[:,i])-hist[dominate,i] # total # of tokens of non-dominating classes
            if remain == 1:
                sbin[i] = 1   # treat it as single

# decide which bin edge can be deleted
# use an action table (atable) to do so. atable is a 3x3 matrix, its entry is an action class
# actions are:
#   	   |void  |single|mix
#-------------------------
# void   |delete|keep  |keep
#----------------------------
# single |keep  |check1|keep
#----------------------------
# mix    |keep  |keep  |check2
#
#   'D'   delete bin_edge[i]
#   'K'   keep bin_edge[i]
#   'C1'  if binmax[i] == binmax[i-1] then DELETE else KEEP
#   'C2'  if allmix and (binmax[i] == binmax[i-1]) then DELETE else KEEP
    atable = np.array([('D','K','K'),('K','C1','K'),('K','K','C2')], dtype = object)
    delete = np.full(nbin, False, dtype = bool) # initialized to keep all edges

#    print("hist", hist)
#    print("BinEdges", BinEdges)
#    print("sbin:", sbin)
#    print("binmax", binmax)
 
    for i in range(1,nbin): # for each edge, decide to KEEP or DELETE
        action = atable[sbin[i-1],sbin[i]]
        #print("action:", action)
        if action == 'D':
            delete[i] = True
        elif action == 'C1':   # 'C1'  if binmax[i] == binmax[i-1] then delete else keep
            if binmax[i] == binmax[i-1]:
                delete[i] = True
        elif action == 'C2':   # 'C2'  if allmix and (binmax[i] == binmax[i-1]) then delete else keep
            if not allmix:
                delete[i] = True
            elif binmax[i] == binmax[i-1]:
                 delete[i] = True  # otherwise, keep

    MergedBin = []
    MergedBin.append(BinEdges[0]) # get the leftmost boundary edge
    for i in range(1,nbin):
        if delete[i] == False: # if action is NOT delete then keep it.
            MergedBin.append(BinEdges[i])
    MergedBin.append(BinEdges[nbin]) # get this rightmost boundary edge
    
# this algo requires that it MUST partition the the feature space into at least 2 parts !!!
# otherwise, it will go to an infinite looop ????
    # special case !!! the following is to break away from the curse !!
    if len(MergedBin) == 2: # only one partition? need to create one more !
        low = BinEdges[0]
        for i in range(0,nbin): # get rid of leading skip bins
            if sbin[i] == 0:
                low = BinEdges[i+1]
            else:
                break
        high = BinEdges[-1]
        for i in range(nbin-1,0,-1):
            if sbin[i] == 0:
                high = BinEdges[i]
            else:
                break
        MergedBin.insert(1, (low+high)/2)    
#        print("MergedBin:", MergedBin)

    partitions = []
    for i in range(1,len(MergedBin)):
        partitions.append([MergedBin[i-1],MergedBin[i]])

#    print("partitions:", partitions)
    return partitions

def DirectSelect(dataset, target, Fminmax, Fmin, Fmax, nF, nC, oFidx, low, high):
    fdata = [[] for i in range(0,nF)]
    sdata = [[] for i in range(0,nF)]
    starget = [[] for i in range(0,nF)]
    div = np.zeros((nF, len(target)+1))
    mark = np.zeros((nF,len(target)-1))
    interval = [[] for i in range(0,nF)]
    nDiv = [1.e6 for i in range(0,nF)]
#    nDiv = np.ones(nF)*1.e6
    
    for i in range(0,nF):
#        if i == oFidx:
#            continue
        fdata[i] = dataset[:,i] # Factor i column
        sdata[i] = np.sort(fdata[i])
        starget[i] = target[np.argsort(fdata[i])]
        for j in range(0,len(sdata[i])-1):
            div[i,j+1] = (sdata[i][j+1] + sdata[i][j])/2
        div[i,0] = sdata[i].min()
        div[i,-1] = sdata[i].max()

        for j in range(0,len(starget[i])-1):
            if starget[i][j] == starget[i][j+1]:
                mark[i,j] = True
        
        interval[i].append(div[i][0])
        for j in range(0,len(dataset)-1):
            if mark[i,j] != True:
                interval[i].append(div[i,j+1])
        
        interval[i].append(div[i,-1])
        nDiv[i] = len(interval[i])
    
    Fidx = np.argmin(nDiv)  # select factor that yields min. # of divisions
    partitions = []
    for i in range(nDiv[Fidx]-1):
        partitions.append([interval[Fidx][i], interval[Fidx][i+1]])
# make sure the 1st partition & the last partition taken from Fminmax
    partitions[0][0]=Fmin[Fidx]
    partitions[0][0]=Fminmax[Fidx,0]
    nP = len(partitions) - 1
    partitions[nP][1]=Fmax[Fidx]
    partitions[nP][1]=Fminmax[Fidx,1]
# !!! below NOT TEST !!!
    if Fidx == oFidx:
            partitions[0][0] = low
            partitions[nP][1]= high

    return Fidx, partitions
    
def write2file(node, file, nF):
    for key, value in (node.state.items()):
        buf = "Class %s : %s ; " % (key, value)
        file.write(buf)
    file.write('\nFminmax:\n')
    
    for i in range(0,nF):
        fmin = node.Fmin[i]
        fmax = node.Fmax[i]
        oFmin, oFmax = node.Fminmax[i,0], node.Fminmax[i,1]
        buf = "Factor[%d] : min= %0.3f ; max = %0.3f" % (i, oFmin, oFmax)
        file.write(buf+'\n')
        
    file.write("partitions for next layer based on oFidx:\n")
    for i in range(0,len(node.partitions)):
        buf="[%0.3f, %0.3f]\n" % (node.partitions[i][0], node.partitions[i][1])
        file.write(buf)
    return
