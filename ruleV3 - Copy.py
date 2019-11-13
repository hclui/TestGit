# RuleV3 -- class rule

class rule:
    def __init__(self, Rid, IdxStr, Fmin, Fmax,  Fset, cls, Fseq, data, target, father):
        self.Rid    = Rid     # rule id, an integer number
        self.IdxStr = IdxStr  # index str indicating the positon of this node on the tree
        self.Fmin   = Fmin    # min value of each feature
        self.Fmax   = Fmax    # max value of each factor
        self.Fset   = Fset    # set of UNIQUE active features for this rule
        self.cls    = cls     # the class this rule belongs to
######        
        self.Fseq   = Fseq    # sequence of features used to arrive at this node
        self.data   = data    # stores the INDEX to training data that falls to this rule
        self.target = target  # stores the corresponding target of training data
        self.father = father  # point back to father node; for debugging
        