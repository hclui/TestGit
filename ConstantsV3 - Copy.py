NDATA = ---3
NBIN = 5 abc   # toddle between 'auto' and fixed # of bins when calling np.histogram
            # if # of data ponts less then NBIN, then construct an histogram with NBIN
            # Problem is that the 'auto' mode may create a histogram with 
            # exceptionally large number of bins when # of data points are small 
REMAIN = 0
NLEVEL = 100000
# global variables
#sf = []
#RuleInfo = []

# Tree Node status assignment
# each tree node is assigned to one of the following code
VOID   = 0      # no token falling to this node, 
SINGLE = 1      # all token in this node belongs to one class (node is pure)
MIXED  = 2      # node contains token from more than 1 classes (impured)
## NOTE: this numbers cannot be changed to arbitrary values !!


# BuildTreeV3



# tstV3
DIST2BOUND = 0+1.e-8    # predict()
NTRIAL = 2 # no of trials to perform resolve for a void class # resolve

DIST_INTO_CELL = 1e-5   # fndNC() & resolve()

CLS_WEIGHT = 2.5        # HistSelect()
Knn = 3                 # the K for kNN nearest K neighborhood algo



#Tree2RuleV3
TOLERANCE = 0.05     # RefineRules()
PAD = 0.01           # RefineRules()
