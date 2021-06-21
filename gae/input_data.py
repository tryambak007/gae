import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd


def load_KG(path):
    keyconcepts = pd.read_csv(path)
    keyconcepts.drop(['sentence', 'subject', 'relation' , 'object', 'subject_flag', 'object_flag'], axis = 1, inplace = True)
    mappings = {}
    inv_map = {}
    idx = 0
    for row in keyconcepts.iterrows():
        subj = row[1]['subject_keys'].strip().lower()
        obj = row[1]['object_keys'].strip().lower()
        if subj not in mappings.keys():
            mappings[subj] = idx
            inv_map[idx] = subj
            idx+=1
        if obj not in mappings.keys():
            mappings[obj] = idx
            inv_map[idx] = obj
            idx += 1
    print("Total nodes : ", idx)
    adjacency = {}
    for row in keyconcepts.iterrows():
        subidx = mappings[row[1]['subject_keys'].strip().lower()]
        objidx = mappings[row[1]['object_keys'].strip().lower()]

        if subidx in adjacency.keys():
            if objidx not in adjacency[subidx]:
                adjacency[subidx].append(objidx)
        else:
            adjacency[subidx] = []
            adjacency[subidx].append(objidx)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adjacency))
    return adj, np.identity(adj.shape[0])



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features
