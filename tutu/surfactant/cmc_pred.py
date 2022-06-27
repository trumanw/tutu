import random

import dgl
from dgl.nn.pytorch import GraphConv
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import mol_to_bigraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem

# Torch setup
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
setup_seed(73)

class GCNReg(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.w1 = nn.Parameter(torch.zeros(1, 1) + 0.5, requires_grad=True)
        self.w2 = nn.Parameter(torch.zeros(1, 1) + 0.1, requires_grad=True)
        self.classify1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

    def forward(self, g):
        h = g.ndata['h'].float()
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.ln1(self.conv1(g, h)))
        h1 = F.relu(self.ln2(self.conv2(g, h1)))
        g.ndata['h'] = h1
        hg = dgl.mean_nodes(g, 'h')
        hg_max = dgl.max_nodes(g, 'h')
        hg = torch.cat([F.normalize(hg, p=2, dim=1), F.normalize(hg_max, p=2, dim=1)], dim=1)
        output = F.relu(self.ln3(self.classify1(hg)))
        output = F.relu(self.ln4(self.classify2(output)))
        output = self.classify3(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output

def process(one_smiles):
    m = Chem.MolFromSmiles(one_smiles)
    node_enc = CanonicalAtomFeaturizer()
    edge_enc = None
    g = mol_to_bigraph(m, True, node_enc, edge_enc, False)
    return g

def predict(one_smiles, model_path):
    if type(one_smiles) == type('a'):
        one_smiles = [one_smiles]
    elif type(one_smiles) == type([0]):
        pass
    else:
        assert TypeError('need one smiles str or smiles list')
    one_smiles = [process(x) for x in one_smiles]
    batched_graph = dgl.batch(one_smiles)
    checkpoint = torch.load(model_path)
    test_model = GCNReg(74, 256, 1, False)
    test_model.eval()
    with torch.no_grad():
        test_model.load_state_dict(checkpoint['state_dict'])
        y_predict = test_model(batched_graph).reshape(-1).detach().numpy().tolist()
    return y_predict