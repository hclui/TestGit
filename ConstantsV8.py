# V8
# this version copies from V31
import numpy as np
# Flags
POSTEDIT    = 1
PRUNE       = 0
FIXBIN      = 1      # ==1 if fixed bin length; else use quantile if many pts in node
DIFF_FACTOR = 1 # each subtree can't use the same factor as father node
GINIFLAG    = 1 # call Gini, else maxerr()
FNDOUTLIERS = 0 # if set to 1, then call FndOutliers()
COMBINEREGION=1 # if set to 1, then call CombineRegion()
GMATOFFSET  = 1
GINIOFFSET   = 0.1 # 0.1653
MAXERROFFSET = 0.0909
# BuildTree()
NLEVEL = 10 #5  #10 # NEEDED in BuildTree()
# OptSplit()
NDIVISION = 11 # 51
FSCORE_WEIGHT = 0.0    # 0.01    # OptSplit()
EPSILON = 1.e-8         # small constant for zero tolerance, OptSplit()
#GiniMatrix()
# PostEdit() parameters
EXT_RANGE = 0.05
OUTTHR = 0.5    # 0.65; if majority cls in range > 70% then delete node 
OUTLIERS = 2
# FndOutlier()  
OUTLIER = 1
DOMINANT_THR = 2.0 #2.0  #  FndOutlier()
# Show graphs
# ShowGraphs == 0    Don't show any graphs;
# ShowGraphs == 1    Show graphs whenever creating new child node
# ShowGraphs == 2    call ShowRectangles in FS_showV8.py to show data, training & testing rectangular
ShowGraphs = 3

##MAX_ERR_THR = 0.05  # 5 tokens out of 100, then change to status=single
## the above approach NOT effective at all. 
# dp penalty
PENALTY = 0.0 #0.00001

#GAIN_THR = 0    # 0.0051  #BuildTree() if Info-gain > Thr, then build subtree
# pruning() 
TOTAL_THR = 2.5 # not used right now
#DOMINANT_THR = 1.0 #0.95
#TOK_CNT =  4
#FWEIGHT = 1


# tstV8 
DIST2BOUND = 0+1.e-8    # Distance to Boundary; FactorSpaceClassifierV8(), tstV8->fNNVR()
Knn = 1                 # tstV8->kNN(); the K for kNN nearest K neighborhood algo

#Tree2RuleV8 -> RefineRules()
TOLERANCE = 0.05     # RefineRules()
PAD = 0.01           # RefineRules()

# Tree Node status assignment
# each tree node is assigned to one of the following code
VOID   = 0      # no token falling to this node, 
SINGLE = 1      # all token in this node belongs to one class (node is pure)
MIXED  = 2      # node contains token from more than 1 classes (impured)
## NOTE: this numbers cannot be changed to arbitrary values !!

###NotSelect  = False       # HistSelect won't select same feature again for child
# NotSelect = True will degrade performance on Cancer data !!
#REMAIN = 0  # NEEDED in GetState()   !!
##constants for sparse implementation
#FminFloor = 1.0e-7
