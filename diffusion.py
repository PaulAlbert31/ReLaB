#From https://github.com/ahmetius/LP-DeepSSL/
import faiss
from faiss import normalize_L2
import torch
import torch.nn.functional as F
import scipy
import numpy as np

#Takes as argument features X, labels for each sample labels, list of labeled samples labeled_idx and return pseudo labels, weights and class weights
def diffusion(X, labels, labeled_idx, k = 100, max_iter = 20, classes=10):
        print('Starting diffusion...')
        alpha = 0.99
        labels = labels.numpy()
        labeled_idx = np.asarray(labeled_idx)
        
        # kNN search for the graph
        d = X.shape[1]
        '''
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatL2(res,d,flat_config)   # build the index FlatL2
        '''
        index = faiss.IndexFlatL2(d)   # build the index FlatL2
        
        normalize_L2(X)
        index.add(X) 
        N = X.shape[0]
        Nidx = index.ntotal

        D, I = index.search(X, k + 1)

        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N,classes))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
        for i in range(classes):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:,i] = f

        # Handle numerical errors
        Z[Z < 0] = 0 

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
        probs_l1[probs_l1 <0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(classes)
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1,1)

        p_labels[labeled_idx] = labels[labeled_idx]
        weights[labeled_idx] = 1.0

        p_weights = weights#.tolist()
        p_labels = p_labels

        # Compute the weight for each class
        class_weights = np.zeros(classes)
        for i in range(classes):
            cur_idx = np.where(np.asarray(p_labels) == i)[0]
            class_weights[i] = (float(labels.shape[0]) / classes) / cur_idx.size

        return probs_l1, p_weights, class_weights
