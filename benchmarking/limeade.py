"""
LIMEADE surrogate model
Local Interpretable Manifold Explanations for Dimension Evaluations, after https://openreview.net/pdf?id=kmLV911L80#page=8.82

Author: Claire He 
23/07/2026
"""
import numpy as np
from sklearn.linear_model import MultiTaskLasso

class LIMEADE:
    """
    LIMEADE: Local Interpretable Manifold Explanations for Dimension Evaluations
    Given original data X (n \times p), reduced data Y (n \times r), penalty coef \lambda >= 0:
    Learn V = \arg\min(0.5 ||Y-XV||_2^2 + \lambda \sum_{j=1}^p ||V_j||_2) 
    
    """
    def __init__(self, data, reduced_data, fit_intercept=True, max_iter=10000, tol=1e-7):
        self.X = np.asarray(data, dtype=float)
        self.Y = np.asarray(reduced_data,dtype=float)
        # check row coherence
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y should have the same number of observations")
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, lambda_reg=1.0, level='global', partition_idx=None, gamma=0.5, debug=False):
        self.level = level
        self.debug = debug
        
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non negative")
        level = str(level).strip().lower()
        
        self.lambda_reg = float(lambda_reg)
        self.partition_idx = partition_idx
        self.gamma = float(gamma)

        if level=='global':
            V = self._global_limeade()
        elif level=='partition':
            V = self._partition_limeade()
        elif level=='local':
            V = self._local_limeade()
        else:
            print("level should be any of 'global','partition' or 'local'")
            
        self.V = V
        return V

    def _global_limeade(self):
        """
        Optimize for \arg\min(0.5 ||Y-XV||_2^2 + \lambda \sum_{j=1}^p ||V_j||_2) 
        """
        V = self._fit_group_lasso(self.X, self.Y)
        return V
        
    def _partition_limeade(self):
        """
        Optimize for \arg\min(0.5 ||Y_{M_k}-X_{M_k}V_k||_2^2 + \lambda \sum_{j=1}^p ||V_j||_2) for each partition M_k
        """
        n, p = self.X.shape
        r = self.Y.shape[1]
        
        if self.partition_idx is None: 
            raise ValueError("partition_idx should be provided")

        labels = np.asarray(self.partition_idx)
        if len(labels) != n:
            raise ValueError("partition_idx should have one label per observation")

        V = {}
        partition_md = {}
        for partition in np.unique(labels):
            mask = (labels == partition)
            partition_size = int(mask.sum())
            V_k = self._fit_group_lasso(self.X[mask], self.Y[mask])
            V[partition] = V_k
            if self.debug:
                partition_md[partition] = self.md.copy()
        self.V = V
        self.partition_md = partition_md
        return V

    def _local_limeade(self):
        """
        Optimize for \arg\min(0.5 ||W^{0.5}(Y-XV)||_2^2 + \lambda \sum_{k=1}^p ||V_j||_2)
        where for row X_i, W_i = Diag(w_i) and w_i = \exp(-\gamma||x_i - X||^2)

        For new point x, use the nearest neighbor's projection \hat v and associate \hat y = x \hat v
        """
        n, p, r = self.X.shape[0], self.X.shape[1], self.Y.shape[1]
        anchor_indices = np.arange(n)
        V = np.empty((n, p, r),dtype=float)

        self.anchor_idx = anchor_indices
        self.X_local = self.X[anchor_indices].copy()
        local_md = {}
        
        for local_idx, anchor_idx in enumerate(anchor_indices):
            d_2 = np.sum((self.X - self.X[anchor_idx])**2, axis=1)
            w = np.exp(-self.gamma * d_2)
            V_k = self._fit_group_lasso(self.X, self.Y, sample_weight=w)
            V[local_idx] = V_k
            if self.debug:
                local_md[local_idx] = self.md.copy()
        self.V=V
        if self.debug:
            self.local_md = local_md
        return V
        

    def _fit_group_lasso(self, X, y, sample_weight=None):
        """ Fits group lasso model
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n, p = X.shape
        _, r = y.shape

        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            sample_weight = np.asarray(sample_weight) 
            if len(sample_weight) != n:
                raise ValueError("sample weight must be one per observation")
            if np.any(sample_weight < 0):
                raise ValueError("weights should be non negative")
                
        pos_weight = sample_weight > 0 
        if pos_weight.sum() < 2:
            raise ValueError("At least two observations should have positive weights")

        X_pos, y_pos = X[pos_weight], y[pos_weight]
        active_weight = sample_weight[pos_weight]
        n_active = X_pos.shape[0]

        if self.fit_intercept:
            x_mean = np.average(X_pos, axis=0, weights=active_weight)
            y_mean = np.average(y_pos, axis=0, weights=active_weight)
        else:
            x_mean = np.zeros(n)
            y_mean = np.zeros(r)

        X_pos, y_pos = X_pos - x_mean, y_pos - y_mean
        scaled_weights = n_active * active_weight
        X_w = np.sqrt(scaled_weights)[:, None] * X_pos
        y_w = np.sqrt(scaled_weights)[:, None] * y_pos 

        # Use MultiTaskLasso for group Lasso
        model = MultiTaskLasso(alpha=self.lambda_reg, fit_intercept=False, max_iter=self.max_iter, tol=self.tol)
        model.fit(X_w, y_w)

        V = model.coef_.T
        if self.fit_intercept: 
            intercept = y_mean - x_mean @ V
        else:
            intercept = np.zeros(r)

        norm_feat = np.linalg.norm(V, axis=1)
        selected_feat = norm_feat > self.tol
        ESS = active_weight.sum()**2/np.sum(active_weight ** 2)

        if self.debug:
            self.md = {"V":V,
                        "intercept":intercept,
                        "norm_feat":norm_feat,
                        "selected_feat":selected_feat,
                        "model":model,
                        "x_mean":x_mean,
                        "y_mean":y_mean,
                        "n_active":n_active,
                        "weight_sum":active_weight.sum(),
                        "ESS":ESS}
            
        return V   
                
    def score(self, anchor=None, partition=None, normalize=False):
        """
        Return feature attributions based on row norms of V.
    
        Global:
            attribution_j = ||V[j, :]||_2
    
        Partition:
            attribution_{k,j} = ||V_k[j, :]||_2
    
        Local:
            attribution_{i,j} = ||V_i[j, :]||_2
        """
        if not hasattr(self, "V"):
            raise RuntimeError("Call fit before requesting attributions.")
    
        if self.level == "global":
            attribution = np.linalg.norm(self.V, axis=1)
    
        elif self.level == "partition":
            if partition is None:
                return {label: np.linalg.norm(V_k, axis=1) for label, V_k in self.V.items()}
            if partition not in self.V:
                raise ValueError(f"No fitted model for partition {partition!r}.")
    
            attribution = np.linalg.norm(self.V[partition], axis=1)
    
        elif self.level == "local":
            if anchor is None:
                attribution = np.linalg.norm(self.V, axis=2)
            else:
                matches = np.flatnonzero(self.anchor_idx == anchor)
                if len(matches) == 0:
                    raise ValueError(f"No local model was fitted at anchor {anchor}.")
                local_idx = matches[0]
                attribution = np.linalg.norm(self.V[local_idx], axis=1)
        else:
            raise RuntimeError(f"Unknown fitted level {self.level!r}.")
    
        if normalize:
            if isinstance(attribution, dict):
                attribution = {key: value / value.sum() if value.sum() > 0 else value for key, value in attribution.items()}
            elif attribution.ndim == 1:
                total = attribution.sum()
                if total > 0:
                    attribution = attribution / total
            else:
                totals = attribution.sum(axis=1, keepdims=True)
                attribution = np.divide(attribution, totals, out=np.zeros_like(attribution), where=totals > 0)
        return attribution   
                
                
            