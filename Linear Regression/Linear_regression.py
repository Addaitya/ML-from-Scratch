import numpy as np
import math

class Linear_regression:
    def __init__(self):
        self.w = None
        self.b = None
        
    def predict(self, X):
        '''
        Computes the output of the model when given model parameters and inputs
        
        Args:
        
        X (ndarray(m, n)): m examples with n features
        w (ndarray(n,)): model parameters(n weights for n features).
        b (scalar): model parameter.
    
        Returns:
            f_wb (ndarray(m, )): m predictions for m input examples
        '''
    
        f_wb = np.matmul(X, self.w.T) + self.b
    
        return f_wb

    def compute_squared_error(self, X, y):
        '''
        Returns squared error when given input, output and model parameters
    
        Args:
            X (ndarray(m, n)): m examples and n features
            y (ndarray(m)): m outputs for m examples
            w (ndarray(n, )): model parameter
            b (scalar): model parameter
    
        Returns:
            cost (scalar): Squared error
        '''
        m = X.shape[0]  #  m is no of examples in X
        f_wb = np.matmul(X, self.w.T) +self.b
        error = (f_wb - y)**2
        cost = np.sum(error) / (2*m)
    
        return cost

    def fit(self, X, y, iterations=10000, *, alpha=0.01, w_in=None, b_in=None):
        m, n = X.shape
        
        w = w_in if w_in else np.zeros(n)
        b = b_in if b_in else 0.0
    
        for _ in range(iterations):
            # compute gradient
            f_wb = np.matmul(X, w.T) + b
            error = f_wb - y
            dJ_dw = np.matmul(X.T, error.T) / m
            dJ_db = np.sum(error) / m
    
            w = w - alpha*dJ_dw
            b = b - alpha*dJ_db

        self.w = w
        self.b = b
        
        return w, b

    def score(self, X, y_true):
        '''
        The function is from sklearn LinearRegression doc
        '''
        y_pred = self.predict(X)

        u = ((y_true - y_pred)** 2).sum() 
        v = ((y_true - y_true.mean()) ** 2).sum()

        sc = 1 - (u/v)
        return sc

    def sample(self):
             """A Bagging regressor.
    
        A Bagging regressor is an ensemble meta-estimator that fits base
        regressors each on random subsets of the original dataset and then
        aggregate their individual predictions (either by voting or by averaging)
        to form a final prediction. Such a meta-estimator can typically be used as
        a way to reduce the variance of a black-box estimator (e.g., a decision
        tree), by introducing randomization into its construction procedure and
        then making an ensemble out of it.
    
        This algorithm encompasses several works from the literature. When random
        subsets of the dataset are drawn as random subsets of the samples, then
        this algorithm is known as Pasting [1]_. If samples are drawn with
        replacement, then the method is known as Bagging [2]_. When random subsets
        of the dataset are drawn as random subsets of the features, then the method
        is known as Random Subspaces [3]_. Finally, when base estimators are built
        on subsets of both samples and features, then the method is known as
        Random Patches [4]_.
    
        Read more in the :ref:`User Guide <bagging>`.
    
        .. versionadded:: 0.15
    
        Parameters
        ----------
        estimator : object, default=None
            The base estimator to fit on random subsets of the dataset.
            If None, then the base estimator is a
            :class:`~sklearn.tree.DecisionTreeRegressor`.
    
            .. versionadded:: 1.2
               `base_estimator` was renamed to `estimator`.
    
        n_estimators : int, default=10
            The number of base estimators in the ensemble.
    
        max_samples : int or float, default=1.0
            The number of samples to draw from X to train each base estimator (with
            replacement by default, see `bootstrap` for more details).
    
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
    
        max_features : int or float, default=1.0
            The number of features to draw from X to train each base estimator (
            without replacement by default, see `bootstrap_features` for more
            details).
    
            - If int, then draw `max_features` features.
            - If float, then draw `max(1, int(max_features * n_features_in_))` features.
    
        bootstrap : bool, default=True
            Whether samples are drawn with replacement. If False, sampling
            without replacement is performed.
    
        bootstrap_features : bool, default=False
            Whether features are drawn with replacement.
    
        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate
            the generalization error. Only available if bootstrap=True.
    
        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble. See :term:`the Glossary <warm_start>`.
    
        n_jobs : int, default=None
            The number of jobs to run in parallel for both :meth:`fit` and
            :meth:`predict`. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors. See :term:`Glossary <n_jobs>` for more details.
    
        random_state : int, RandomState instance or None, default=None
            Controls the random resampling of the original dataset
            (sample wise and feature wise).
            If the base estimator accepts a `random_state` attribute, a different
            seed is generated for each instance in the ensemble.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
    
        verbose : int, default=0
            Controls the verbosity when fitting and predicting.
    
        Attributes
        ----------
        estimator_ : estimator
            The base estimator from which the ensemble is grown.
    
            .. versionadded:: 1.2
               `base_estimator_` was renamed to `estimator_`.
    
        n_features_in_ : int
            Number of features seen during :term:`fit`.
    
            .. versionadded:: 0.24
    
        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.
    
            .. versionadded:: 1.0
    
        estimators_ : list of estimators
            The collection of fitted sub-estimators.
    
        estimators_samples_ : list of arrays
            The subset of drawn samples (i.e., the in-bag samples) for each base
            estimator. Each subset is defined by an array of the indices selected.
    
        estimators_features_ : list of arrays
            The subset of drawn features for each base estimator.
    
        oob_score_ : float
            Score of the training dataset obtained using an out-of-bag estimate.
            This attribute exists only when ``oob_score`` is True.
    
        oob_prediction_ : ndarray of shape (n_samples,)
            Prediction computed with out-of-bag estimate on the training
            set. If n_estimators is small it might be possible that a data point
            was never left out during the bootstrap. In this case,
            `oob_prediction_` might contain NaN. This attribute exists only
            when ``oob_score`` is True.
    
        See Also
        --------
        BaggingClassifier : A Bagging classifier.
    
        References
        ----------
    
        .. [1] L. Breiman, "Pasting small votes for classification in large
               databases and on-line", Machine Learning, 36(1), 85-103, 1999.
    
        .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
               1996.
    
        .. [3] T. Ho, "The random subspace method for constructing decision
               forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
               1998.
    
        .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
               Learning and Knowledge Discovery in Databases, 346-361, 2012.
    
        Examples
        --------
        >>> from sklearn.svm import SVR
        >>> from sklearn.ensemble import BaggingRegressor
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=4,
        ...                        n_informative=2, n_targets=1,
        ...                        random_state=0, shuffle=False)
        >>> regr = BaggingRegressor(estimator=SVR(),
        ...                         n_estimators=10, random_state=0).fit(X, y)
        >>> regr.predict([[0, 0, 0, 0]])
        array([-2.8720...])
        """
        pass