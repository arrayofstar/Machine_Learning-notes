# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/055_models.MINIROCKET.ipynb.

# %% auto 0
__all__ = ['MiniRocketClassifier', 'load_minirocket', 'MiniRocketRegressor', 'MiniRocketVotingClassifier', 'get_minirocket_preds',
           'MiniRocketVotingRegressor']

# %% ../../nbs/055_models.MINIROCKET.ipynb 3
import sklearn
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from ..imports import *


# %% ../../nbs/055_models.MINIROCKET.ipynb 4
class MiniRocketClassifier(sklearn.pipeline.Pipeline):
    """Time series classification using MINIROCKET features and a linear classifier"""
    def __init__(self, num_features=10_000, max_dilations_per_kernel=32, random_state=None,
                 alphas=np.logspace(-3, 3, 7), normalize_features=True, memory=None, verbose=False, scoring=None, class_weight=None, **kwargs):
        """ MiniRocketClassifier is recommended for up to 10k time series. 
        
        For a larger dataset, you can use MINIROCKET (in Pytorch).
        scoring = None --> defaults to accuracy.
        """
        
        try: 
            import sktime
            from sktime.transformations.panel.rocket._minirocket_multivariate import MiniRocketMultivariate
        except ImportError: 
            raise ImportError("You need to install sktime to be able to use MiniRocketClassifier")
            
        self.steps = [('minirocketmultivariate', MiniRocketMultivariate(num_kernels=num_features, 
                                                                        max_dilations_per_kernel=max_dilations_per_kernel,
                                                                        random_state=random_state))]
        if normalize_features:
            self.steps += [('scalar', StandardScaler(with_mean=False))]
        
        self.steps += [('ridgeclassifiercv', RidgeClassifierCV(alphas=alphas, 
                                                              scoring=scoring, 
                                                              class_weight=class_weight, 
                                                              **kwargs))]
        store_attr()
        self._validate_steps()

    def __repr__(self):
        return f'Pipeline(steps={self.steps.copy()})'

    def save(self, fname=None, path='./models'):
        fname = ifnone(fname, 'MiniRocketClassifier')
        path = Path(path)
        filename = path/fname
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(f'{filename}.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

# %% ../../nbs/055_models.MINIROCKET.ipynb 5
def load_minirocket(fname, path='./models'):
    path = Path(path)
    filename = path/fname
    with open(f'{filename}.pkl', 'rb') as input:
        output = pickle.load(input)
    return output

# %% ../../nbs/055_models.MINIROCKET.ipynb 6
class MiniRocketRegressor(sklearn.pipeline.Pipeline):
    """Time series regression using MINIROCKET features and a linear regressor"""
    def __init__(self, num_features=10000, max_dilations_per_kernel=32, random_state=None,
                 alphas=np.logspace(-3, 3, 7), *, normalize_features=True, memory=None, verbose=False, scoring=None, **kwargs):
        """ MiniRocketRegressor is recommended for up to 10k time series. 
        
        For a larger dataset, you can use MINIROCKET (in Pytorch).
        scoring = None --> defaults to r2.
        """
        
        try: 
            import sktime
            from sktime.transformations.panel.rocket._minirocket_multivariate import MiniRocketMultivariate
        except ImportError: 
            raise ImportError("You need to install sktime to be able to use MiniRocketRegressor")
            
        self.steps = [('minirocketmultivariate', MiniRocketMultivariate(num_kernels=num_features,
                                                                        max_dilations_per_kernel=max_dilations_per_kernel,
                                                                        random_state=random_state))]
        if normalize_features:
            self.steps += [('scalar', StandardScaler(with_mean=False))]
        
        self.steps += [('ridgecv', RidgeCV(alphas=alphas, scoring=scoring, **kwargs))]
        store_attr()
        self._validate_steps()

    def __repr__(self):
        return f'Pipeline(steps={self.steps.copy()})'

    def save(self, fname=None, path='./models'):
        fname = ifnone(fname, 'MiniRocketRegressor')
        path = Path(path)
        filename = path/fname
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(f'{filename}.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

# %% ../../nbs/055_models.MINIROCKET.ipynb 7
def load_minirocket(fname, path='./models'):
    path = Path(path)
    filename = path/fname
    with open(f'{filename}.pkl', 'rb') as input:
        output = pickle.load(input)
    return output

# %% ../../nbs/055_models.MINIROCKET.ipynb 8
class MiniRocketVotingClassifier(VotingClassifier):
    """Time series classification ensemble using MINIROCKET features, a linear classifier and majority voting"""
    def __init__(self, n_estimators=5, weights=None, n_jobs=-1, num_features=10_000, max_dilations_per_kernel=32, random_state=None, 
                 alphas=np.logspace(-3, 3, 7), normalize_features=True, memory=None, verbose=False, scoring=None, class_weight=None, **kwargs):
        store_attr()
        
        try: 
            import sktime
        except ImportError: 
            raise ImportError("You need to install sktime to be able to use MiniRocketVotingClassifier")
            
        estimators = [(f'est_{i}', MiniRocketClassifier(num_features=num_features, max_dilations_per_kernel=max_dilations_per_kernel, 
                                                       random_state=random_state, alphas=alphas, normalize_features=normalize_features, memory=memory, 
                                                       verbose=verbose, scoring=scoring, class_weight=class_weight, **kwargs)) 
                    for i in range(n_estimators)]
        super().__init__(estimators, voting='hard', weights=weights, n_jobs=n_jobs, verbose=verbose)

    def __repr__(self):   
        return f'MiniRocketVotingClassifier(n_estimators={self.n_estimators}, \nsteps={self.estimators[0][1].steps})'

    def save(self, fname=None, path='./models'):
        fname = ifnone(fname, 'MiniRocketVotingClassifier')
        path = Path(path)
        filename = path/fname
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(f'{filename}.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

# %% ../../nbs/055_models.MINIROCKET.ipynb 9
def get_minirocket_preds(X, fname, path='./models', model=None):
    if X.ndim == 1: X = X[np.newaxis][np.newaxis]
    elif X.ndim == 2: X = X[np.newaxis]
    if model is None: 
        model = load_minirocket(fname=fname, path=path)
    return model.predict(X)

# %% ../../nbs/055_models.MINIROCKET.ipynb 10
class MiniRocketVotingRegressor(VotingRegressor):
    """Time series regression ensemble using MINIROCKET features, a linear regressor and a voting regressor"""
    def __init__(self, n_estimators=5, weights=None, n_jobs=-1, num_features=10_000, max_dilations_per_kernel=32, random_state=None,
                 alphas=np.logspace(-3, 3, 7), normalize_features=True, memory=None, verbose=False, scoring=None, **kwargs):
        store_attr()
        
        try: 
            import sktime
        except ImportError: 
            raise ImportError("You need to install sktime to be able to use MiniRocketVotingRegressor")
            
        estimators = [(f'est_{i}', MiniRocketRegressor(num_features=num_features, max_dilations_per_kernel=max_dilations_per_kernel,
                                                      random_state=random_state, alphas=alphas, normalize_features=normalize_features, memory=memory,
                                                      verbose=verbose, scoring=scoring, **kwargs))
                      for i in range(n_estimators)]
        super().__init__(estimators, weights=weights, n_jobs=n_jobs, verbose=verbose)

    def __repr__(self):
        return f'MiniRocketVotingRegressor(n_estimators={self.n_estimators}, \nsteps={self.estimators[0][1].steps})'

    def save(self, fname=None, path='./models'):
        fname = ifnone(fname, 'MiniRocketVotingRegressor')
        path = Path(path)
        filename = path/fname
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(f'{filename}.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
