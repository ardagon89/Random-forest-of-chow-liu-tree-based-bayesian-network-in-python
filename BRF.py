#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import random
import copy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def loadfile(filename1, filename2=None):
    ds1 = np.loadtxt(filename1, delimiter=",", dtype=int)
    if filename2:
        ds2 = np.loadtxt(filename1, delimiter=",", dtype=int)
        ds = np.vstack((ds1, ds2))
    else:
        ds = ds1
    return ds, ds.shape[0], ds.shape[1]

def generate_bootstrap_data(ds, m, n, k):
    rows = int(0.67*m)
    bds = np.zeros((rows, n, k))
    for i in range(k):
        random_rows = random.sample(range(m), rows)
        bds[:, :, i] = ds[random_rows,:]
    return bds

def prob_matrix(ds, m, n):
    prob_xy = np.zeros((n, n, 4))
    for i in range(n):
        subds = ds[ds[:, i] == 0]
        for j in range(n):
            if prob_xy[i, j, 0] == 0:
                prob_xy[i, j, 0] = (subds[subds[:, j] == 0].shape[0]+1)/(m+4)
            if prob_xy[j, i, 0] == 0:
                prob_xy[j, i, 0] = prob_xy[i, j, 0]
            if prob_xy[i, j, 1] == 0:
                prob_xy[i, j, 1] = (subds[subds[:, j] == 1].shape[0]+1)/(m+4)
            if prob_xy[j, i, 2] == 0:
                prob_xy[j, i, 2] = prob_xy[i, j, 1]
            
        subds = ds[ds[:, i] == 1]
        for j in range(n):
            if prob_xy[i, j, 2] == 0:
                prob_xy[i, j, 2] = (subds[subds[:, j] == 0].shape[0]+1)/(m+4)
            if prob_xy[j, i, 1] == 0:
                prob_xy[j, i, 1] = prob_xy[i, j, 2]
            if prob_xy[i, j, 3] == 0:
                prob_xy[i, j, 3] = (subds[subds[:, j] == 1].shape[0]+1)/(m+4)
            if prob_xy[j, i, 3] == 0:
                prob_xy[j, i, 3] = prob_xy[i, j, 3]
    return prob_xy

def mutual_info(prob_xy, n):
    I_xy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                I_xy[i, j] = prob_xy[i, j, 0]*np.log(prob_xy[i, j, 0]/(prob_xy[i, i, 0]*prob_xy[j, j, 0]))                 + prob_xy[i, j, 1]*np.log(prob_xy[i, j, 1]/(prob_xy[i, i, 0]*prob_xy[j, j, 3]))                 + prob_xy[i, j, 2]*np.log(prob_xy[i, j, 2]/(prob_xy[i, i, 3]*prob_xy[j, j, 0]))                 + prob_xy[i, j, 3]*np.log(prob_xy[i, j, 3]/(prob_xy[i, i, 3]*prob_xy[j, j, 3]))
    return I_xy

def draw_tree(edge_wts, prnt = False, k=0, step=0):
    edge_wts_cp = 1/copy.deepcopy(edge_wts)
    X = csr_matrix(edge_wts_cp)
    Tcsr = minimum_spanning_tree(X)
    edges1 = [(item[0],item[1]) for item in np.transpose(np.nonzero(Tcsr.toarray()))]
    new_tree1 = []
    make_tree(edges1, new_tree1, edges1[0][0])
    return new_tree1

def draw_tree_old(edge_wts, prnt = False):
    edge_wts_cp = copy.deepcopy(edge_wts)
    edges = [np.unravel_index(np.argmax(edge_wts_cp), edge_wts_cp.shape)]
    visited = [[edges[-1][0],edges[-1][1]]]
    edge_wts_cp[edges[-1]] = 0
    while(len(edges) < edge_wts.shape[0]-1):
        i = j = -1
        edge = np.unravel_index(np.argmax(edge_wts_cp), edge_wts_cp.shape)
        for bag in range(len(visited)):
            if edge[0] in visited[bag]:
                i = bag
            if edge[1] in visited[bag]:
                j = bag
        if i == -1 and j != -1:
            edges.append(edge)
            visited[j].append(edge[0])
        elif i != -1 and j == -1:
            edges.append(edge)
            visited[i].append(edge[1])
        elif i == -1 and j == -1:
            edges.append(edge)
            visited.append([edge[0], edge[1]])
        elif i != -1 and j != -1 and i != j:
            edges.append(edge)
            visited[i] += visited[j]
            visited.remove(visited[j])
        elif i == j != -1:
            pass
        else:
            print("Discarded in else", edge)
        edge_wts_cp[edge] = 0

    new_tree = []
    make_tree(edges, new_tree, edges[0][0])
    
    return new_tree

def remove_edges(info_matrix, r):
    count = 0
    while(count<r):
        x,y = random.randint(0,info_matrix.shape[0]-1), random.randint(0,info_matrix.shape[0]-1)
        if x<y and info_matrix[x, y] != 0:
            info_matrix[x, y] = 0
            count += 1
    return info_matrix

def count_matrix(ds, tree, cols):
    count_xy = np.zeros((len(tree), cols))
    for idx, node in enumerate(tree):
        i, j = node
        count_xy[idx] = [ds[(ds[:, i]==0) & (ds[:, j]==0)].shape[0], ds[(ds[:, i]==0) & (ds[:, j]==1)].shape[0], ds[(ds[:, i]==1) & (ds[:, j]==0)].shape[0], ds[(ds[:, i]==1) & (ds[:, j]==1)].shape[0]]

    return count_xy

def make_tree(ls, new_tree, parent):
    for node in [item for item in ls if parent in item]:
        if node[0] == parent:
            new_tree.append(node)
            ls.remove(node)
            make_tree(ls, new_tree, node[1])
        else:
            new_tree.append((node[1],node[0]))
            ls.remove(node)
            make_tree(ls, new_tree, node[0])
            
def sum_mutual_info(tree,I_xy):
    total = 0
    for node in tree:
        if node[0] < node[1]:
            total += I_xy[node[0], node[1]]
        else:
            total += I_xy[node[1], node[0]]
    return total

if __name__ == "__main__":
    import sys

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    
    if len(sys.argv) != 6:
        print("Usage:python BRF.py <k> <r> <training-dataset> <validation-dataset> <testing-dataset>")
    else:
        orig_ds, orig_m, orig_n = loadfile(sys.argv[3],sys.argv[4])
        tv = np.loadtxt(sys.argv[4], delimiter=",", dtype=int)
        ts = np.loadtxt(sys.argv[5], delimiter=",", dtype=int)
        k = int(sys.argv[1])
        fraction_edges_removed = float(sys.argv[2])
        r = int((orig_n*orig_n/2)*fraction_edges_removed)
        base_avg = []
        weighted_avg = []
        for iteration in range(10):
            bds = generate_bootstrap_data(orig_ds, orig_m, orig_n, k)
            weight = [0]*k
            LL = [0]*k
            for i in range(k):
                ds, m, n = bds[:, :, i], bds[:, :, i].shape[0], bds[:, :, i].shape[1]
                prob_xy = prob_matrix(ds, m, n)
                I_xy = mutual_info(prob_xy, n)
                I_xy = remove_edges(I_xy, r)
                tree = draw_tree(I_xy, False)
                tree = [(tree[0][0], tree[0][0])] + tree
                cond_prob = np.zeros((len(tree), prob_xy.shape[2]))
                for idx, node in enumerate(tree):
                    if node[0] == node[1]:
                        cond_prob[idx] = np.log(prob_xy[node[0], node[1],:])
                    else:
                        cond_prob[idx] = np.log(np.hstack(((prob_xy[node[0], node[1],:2]/prob_xy[node[0], node[0], 0]),(prob_xy[node[0], node[1],2:]/prob_xy[node[0], node[0], 3]))))
                count_xy = count_matrix(tv, tree, prob_xy.shape[2])
                weight[i] = np.sum(count_xy*cond_prob)/tv.shape[0]
                count_xy = count_matrix(ts, tree, prob_xy.shape[2])
                LL[i] = np.sum(count_xy*cond_prob)/ts.shape[0]
            weight = np.exp(weight)/np.exp(weight).sum()
            base_avg.append((np.sum(LL) / k)) 
            weighted_avg.append((np.sum(weight * LL)))
        print("k="+str(k)+" edges_removed="+str(r)+" bs_M="+str(np.mean(base_avg))+" wt_M="+str(np.mean(weighted_avg))+" wt_S="+str(np.std(weighted_avg)))

