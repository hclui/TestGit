# Tree2RuleV3

import math
import numpy as np
import copy
from scipy.spatial import distance

from ConstantsV3 import NLEVEL, TOLERANCE, PAD
from ruleV3 import rule


def	Tree2Rules(TreeRoot, nF, nC):
# !! need min & max range for each factor !!!
# this verion uses rule class
    
    queue = []
    lvl = 0
    k = 0
    TreeIdx = [0]
    nRule = 0
#    RuleInfo = []
    Rules = []
    lowbound = np.zeros(nF)
    upbound  = np.zeros(nF)
    Glbminmax = TreeRoot.Fminmax
    # since max range of training data may not be cover the actual mrange
    # the following code is to put a pad to extend the global min & max ranges
    # 
    for i in range(0,nF):
        gmin, gmax = Glbminmax[i,0], Glbminmax[i,1]
        lowbound[i] = gmin
        upbound[i]  = gmax

    queue.append([TreeRoot, lvl, TreeIdx])
    while len(queue) != 0:
        tmp = queue[0]
        del queue[0]        # dequeue
        node = tmp[0].child
        if node is None:
            continue
        else:
            lvl = tmp[1] + 1    # get current tree level & bump it by 1
            TreeIdx = tmp[2]
            k=0
            TreeIdx.append(k)

        while True:
            state = node.state
            status = state.get("status")
            if (status != 'mix') or (status == 'mix' and lvl == NLEVEL):
                IdxStr = '.'.join(map(str,TreeIdx)) 
                Fminmax = node.Fminmax
                Fmin = node.Fmin
                Fmax = node.Fmax
                for i in range(0,nF): # this for loop seems to be useless !!! should delete !!!
                    if Fminmax[i,0] == Glbminmax[i,0] :
                        Fminmax[i,0] = lowbound[i]
                        Fmin[i] = lowbound[i]
                    if Fminmax[i,1] == Glbminmax[i,1] :
                        Fminmax[i,1] = upbound[i]
                    if Fmin[i] == Glbminmax[i,0] :
                        Fmin[i] = lowbound[i]
                    if Fmax[i] == Glbminmax[i,1] :
                        Fmax[i] = upbound[i]
                
                for i in (node.Fset):
                    if  i != -1:
                        print("Rule: %d ; Tree Index: %s ::  %f <= factor[%d] < high: %f " % 
                          (nRule, IdxStr, Fminmax[i,0], i, Fminmax[i,1]))
                print("    state:", state)
                cls = state["cls"]
                dataset = node.dataset
                target  = node.target
                father  = node.father
                Fset    = node.Fset
                Fseq    = node.Fseq

#                RuleInfo.append([nRule, IdxStr, Fminmax, cls, dataset, target, father, Fset])
                Rules.append(rule(nRule, IdxStr, Fmin, Fmax, Fset, cls, Fseq, dataset, target, father))


                nRule += 1

            queue.append([node, lvl, copy.deepcopy(TreeIdx)])
            if node.next is not None:
                node = node.next
                k += 1
                del TreeIdx[-1]
                TreeIdx.append(k)
            else:
                break

#    return RuleInfo
    return Rules

def RefineRules(Rules, nF):
    for Ridx in range(len(Rules)):
        if Rules[Ridx].cls != -1 :  # not void cell
#            Fmin = copy.deepcopy(Rules[Ridx].Fmin)
#            Fmax = copy.deepcopy(Rules[Ridx].Fmax)
            father = copy.deepcopy(Rules[Ridx].father)
            Fset = copy.deepcopy(Rules[Ridx].Fset)
            Fseq = copy.deepcopy(Rules[Ridx].Fseq)
            Rid = len(Rules)
            data = Rules[Ridx].data
            for i in Rules[Ridx].Fset:    # only for active features used in this node
                if i != -1:
                    Fmin = copy.deepcopy(Rules[Ridx].Fmin)
                    Fmax = copy.deepcopy(Rules[Ridx].Fmax)
                    fmin = Fmin[i]
                    fmax = Fmax[i]
                    amin = data[:,i].min()
                    amax = data[:,i].max()
                    if amin > fmin + TOLERANCE: # actual min bigger than fmin, 
                        Rules[Ridx].Fmin[i] = amin - PAD # so shrink fmin of current rule
                        tmp = copy.deepcopy(Fmax)
                        tmp[i] = amin - PAD  # set upper boundary of new rule to be created
                        Rules.append(rule(Rid,0,Fmin,tmp,Fset,-1,Fseq, [],[], father))
                    if amax < fmax - TOLERANCE: # actual max smaller than fmax, 
                        Rules[Ridx].Fmax[i] = amax + PAD # so shrink fmax of current rule
                        tmp = copy.deepcopy(Fmin)
                        tmp[i] = amax + PAD  # set upper boundary of new rule to be created
                        Rules.append(rule(Rid,0,tmp,Fmax,Fset,-1,Fseq, [],[], father))
    return

#def RefineRules(RuleInfo, nF):
#    for Ridx in range(len(RuleInfo)):
#        if RuleInfo[Ridx][3] != -1 :  # not void cell
#            Fminmax = copy.deepcopy(RuleInfo[Ridx][2])
#            dataset = RuleInfo[Ridx][4]
#            #for i in range(nF):    
#            for i in RuleInfo[Ridx][7]:    # only for active features used in this node
#                if i != -1:
#                    fmin, fmax = Fminmax[i][0], Fminmax[i][1]
#                    amin = dataset[:,i].min()
#                    amax = dataset[:,i].max()
#                    if amin > fmin + TOLERANCE:
#                        MkVoidCell(RuleInfo, Ridx,Fminmax, i,0,amin-PAD)
#                    if amax < fmax - TOLERANCE:
#                        MkVoidCell(RuleInfo, Ridx,Fminmax, i,1,amax+PAD)
#
#    return

#def MkVoidCell(RuleInfo, Ridx,Fminmax, i,j,val):
#    nRule = len(RuleInfo)
#    RuleInfo[Ridx][2][i][j] = val # shrink cell of old rule
#    nFminmax = copy.deepcopy(Fminmax)
#    if j == 0:
#        nFminmax[i][1] = val
#    else:
#        nFminmax[i][0] = val
#    RuleInfo.append([nRule,0,nFminmax,-1,[],[],RuleInfo[Ridx][6]])
#    return
#

####################################################################
#def CreateRules(RuleInfo, nF):
#    file = open("FS_Rules.py", "w")
#
#    file.write("def predict(F):"+'\n')
#    file.write("    result = -2" + '\n')
#    
#    for i in range(0,len(RuleInfo)):
#        str1 = '# Rule' + str(i) +', ' + RuleInfo[i][1]  # Rule # & Idxstr
#        file.write(str1 + '\n')
#        Fminmax = RuleInfo[i][2]
#        strbeg = '    if '+str(Fminmax[0,0])+' <= '+'F[0]'+' and '+ 'F[0]'+' < ' + str(Fminmax[0,1])+' and\\'
#        file.write(strbeg + '\n')
#        for j in range(1,nF-1):
#            strmid = '    '+str(Fminmax[j,0])+' <= '+'F['+str(j)+']'+' and '+ 'F['+str(j)+']'+' < ' + str(Fminmax[j,1])+' and\\'
#            file.write(strmid + '\n')
#            
#        strend = '    '+str(Fminmax[nF-1,0])+' <= '+'F['+str(nF-1)+']'+' and '+ 'F['+str(nF-1)+']'+' < ' + str(Fminmax[nF-1,1])+' :'
#        file.write(strend + '\n')
#        
#        strthen = '        result = ' + str(RuleInfo[i][3]) # then predict = class number
#        file.write(strthen + '\n')
#        strthen = '        return result'
#        file.write(strthen + '\n')
#
#    file.write("\n    return result"+'\n')
#    file.close()
#    
#    return

