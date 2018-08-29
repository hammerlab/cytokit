import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import optimize


def get_binary_indicators(n_feature):
        return np.array(np.meshgrid(*[[0, 1] for _ in range(n_feature)])).T.reshape(-1, n_feature)


def get_feature_labels(feature_names, positive='+', negative='-', sep='/'):
    return [
        sep.join([feature_names[i] + (positive if flag > 0 else negative) for i, flag in enumerate(icombo)])
        for icombo in get_binary_indicators(len(feature_names))
    ]


class BinDistOptim(object):
    
    def __init__(self, scale=1.):
        self.scale = scale

    def fit(self, X, y, w0=None, bounds=None):
        assert X.ndim == 2
        X, y = np.array(X), np.array(y)
        nr, nc = X.shape
        assert nc ** 2 == len(y)
        Xmean, Xsd = X.mean(axis=0), X.std(axis=0)
        Xs = (X - Xmean) / Xsd
        
        # Create array of shape (-1, nc) containing all possible binary combinations
        # of each of the given columns to find a threshold for
        idx = get_binary_indicators(nc)
        
        losses = []
        ws = []
        
        def evaluate(w):
            p1 = expit((Xs - w) * self.scale)
            p0 = 1. - p1
            y_est = []
            # Create individual column estimates where each represents one possible binary combo
            for icombo in idx:
                # Build matrix containing per-observation probabilities 
                ps = np.column_stack([(p1[:, i] if flag > 0 else p0[:, i]) for i, flag in enumerate(icombo)])
                
                # Get single per-observation estimate of probability of this combination
                y_est.append(ps.prod(axis=1))
            y_est = np.column_stack(y_est)
            
            # Renormalize per-observation probabilities to sum to 1
            y_est = (y_est / y_est.sum(axis=1).reshape(-1, 1))
            assert np.isclose(y_est.sum(), nr)
            
            # Compute population proportion estimate for each combination of features (and validate shape)
            y_est = y_est.mean(axis=0)
            assert np.isclose(y_est.sum(), 1.)
            assert y_est.ndim == 1
            assert len(y_est) == len(y)
            
            # Compute loss
            # print(y, y_est)
            loss = np.linalg.norm(y - y_est)
            losses.append(loss)
            ws.append(w)
            return y_est, loss
        
        def obj_fn(w):
            return evaluate(w)[1]
            
        if w0 is None:
            w0 = [0. for _ in range(nc)]
        self.optim_res_ = optimize.fmin_slsqp(obj_fn, w0, iter=100, full_output=1, bounds=bounds)
        self.coef_scaled_, self.obj_val_, self.niter_, self.status_code_, _ = self.optim_res_
        self.loss_ = losses
        self.coef_history_ = ws
        
        self.proportions_ = evaluate(self.coef_scaled_)[0]
        
        if self.status_code_ > 0:
            raise ValueError('Optimization failed; Optimizer result = {}'.format(optim_res))
        
        assert self.coef_scaled_.shape == Xsd.shape == Xmean.shape == (nc,)
        self.coef_ = (self.coef_scaled_ * Xsd) + Xmean
        return self