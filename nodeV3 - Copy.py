# nodeV3 -- class node

class node:
    def __init__(self, Fidx, Fseq, Fset, partitions, Fminmax, Fmin, Fmax, state, dataset, target):
        self.Fidx = Fidx
        self.Fseq = Fseq  # list of features used to creat this node
        self.Fset = Fset  # set of UNIQUE features used to get to this node
        self.partitions = partitions
        self.Fminmax = Fminmax    # min max value of each factor for this node
        self.Fmin = Fmin
        self.Fmax = Fmax
        self.state = state # a dictionary of (nC+1) pairs.
        # for each class, stores # of tokens occupying this class; last key is 'status' -- either 'void', 'single' or 'mix'
        self.next = None	# its siblings
        self.child = None	# its child
        self.father = None	# point back to father node; for debugging
        # for building tree
        self.dataset = dataset
        self.target  = target
        
