import numpy as np


from menpo.math import pca


class IRLRegression(object):
    r"""
    Incremental Regularized Linear Regression
    """
    def __init__(self, alpha=0, bias=True):
        self.alpha = alpha
        self.bias = bias
        self.V = None
        self.W = None

    def train(self, X, Y):
        if self.bias:
            # add bias
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # regularized linear regression
        XX = X.T.dot(X)
        # ensure covariance is perfectly symmetric for inversion
        XX = (XX + XX.T) / 2.0
        np.fill_diagonal(XX, self.alpha + np.diag(XX))
        self.V = np.linalg.inv(XX)
        self.W = self.V.dot(X.T.dot(Y))

    def increment(self, X, Y):
        if self.bias:
            # add bias
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # incremental regularized linear regression
        U = X.dot(self.V).dot(X.T)
        np.fill_diagonal(U, 1 + np.diag(U))
        U = np.linalg.inv(U)
        Q = self.V.dot(X.T).dot(U).dot(X)
        self.V = self.V - Q.dot(self.V)
        self.W = self.W - Q.dot(self.W) + self.V.dot(X.T.dot(Y))

    def predict(self, x):
        if self.bias:
            if len(x.shape) == 1:
                x = np.hstack((x, np.ones(1)))
            else:
                x = np.hstack((x, np.ones((x.shape[0], 1))))
        return np.dot(x, self.W)


# TODO: document me!
class IIRLRegression(IRLRegression):
    r"""
    Indirect Incremental Regularized Linear Regression
    """
    def __init__(self, alpha=0, bias=False, alpha2=0):
        # TODO: Can we model the bias? May need to slice off of prediction?
        super(IIRLRegression, self).__init__(alpha=alpha, bias=False)
        self.alpha2 = alpha2

    def train(self, X, Y):
        # regularized linear regression exchanging the roles of X and Y
        super(IIRLRegression, self).train(Y, X)
        J = self.W
        # solve the original problem by computing the pseudo-inverse of the
        # previous solution
        # Note that everything is transposed from the above exchanging of roles
        H = J.dot(J.T)
        H = (H + H.T) / 2.0
        np.fill_diagonal(H, self.alpha2 + np.diag(H))
        self.W = np.linalg.solve(H, J)

    def increment(self, X, Y):
        # incremental least squares exchanging the roles of X and Y
        super(IIRLRegression, self).increment(Y, X)
        J = self.W
        # solve the original problem by computing the pseudo-inverse of the
        # previous solution
        # Note that everything is transposed from the above exchanging of roles
        H = J.dot(J.T)
        np.fill_diagonal(H, self.alpha2 + np.diag(H))
        self.W = np.linalg.solve(H, J)

    def predict(self, x):
        return self.W.dot(x.T).T


class PCRRegression(object):
    r"""
    Multivariate Linear Regression using Principal Component Regression

    Parameters
    ----------
    X : numpy.array
        The regression features used to create the coefficient matrix.
    T : numpy.array
        The shapes differential that denote the dependent variable.
    variance: float or None, Optional
        The SVD variance.
        Default: None

    Raises
    ------
    ValueError
        variance must be set to a number between 0 and 1
    """
    def __init__(self, variance=None, normalise_x=True, bias=True,
                 eps=1e-10):
        self.variance = variance
        self.normalise_x = normalise_x
        self.bias = bias
        self.R = None
        self.eps = eps

    @staticmethod
    def _normalise_x(x):
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0)
        return (x - mean_x) / std_x

    def train(self, X, Y):
        if self.normalise_x:
            X = self._normalise_x(X)

        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Reduce variance of X
        U, l, m = pca(X)
        variation = np.cumsum(l) / np.sum(l)
        if self.variance is not None:
            # Inverted for easier parameter semantics
            k = np.sum(variation < self.variance)
            U = U[:k, :]
        params = (X - m).dot(U.T)
        X_trunc = params.dot(U) + m

        # Perform PCR
        U, l, _ = pca(X_trunc, centre=False, inplace=True)
        inv_eig = np.linalg.inv(np.diag(l))
        cov_inv = U.T.dot(inv_eig.dot(U))
        self.R = cov_inv.dot(X_trunc.T.dot(Y))

    def increment(self, X, Y):
        raise NotImplementedError()

    def predict(self, x):
        if self.normalise_x:
            x = self._normalise_x(x)

        if self.bias:
            if len(x.shape) == 1:
                x = np.hstack((x, np.ones(1)))
            else:
                x = np.hstack((x, np.ones((x.shape[0], 1))))
        return np.dot(x, self.R)


class CCARegression(object):
    r"""
    Multivariate Linear Regression using CCA reconstructions

    Parameters
    ----------
    X : numpy.array
        The regression features used to create the coefficient matrix.
    T : numpy.array
        The shapes differential that denote the dependent variable.
    variance: float or None, Optional
        The SVD variance.
        Default: None

    Raises
    ------
    ValueError
        variance must be set to a number between 0 and 1
    """
    def __init__(self, variance=None, bias=True, eps=1e-10):
        self.variance = variance
        self.bias = bias
        self.R = None
        self.eps = eps

    def train(self, X, Y):
        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Reduce variance of X
        U, l, _ = pca(X, centre=False)
        variation = np.cumsum(l) / np.sum(l)
        k = U.shape[0]
        if self.variance is not None:
            # Inverted for easier parameter semantics
            k = np.sum(variation < self.variance)
            U = U[:k, :]

        inv_eig = np.sqrt(np.linalg.inv(np.diag(l[:k])))
        U = inv_eig.dot(U)

        A = X.T.dot(Y).dot(Y.T).dot(X)
        A_tilde = U.dot(A).dot(U.T)

        V, l2, _ = pca(A_tilde, centre=False)
        H = V.dot(U)

        self.R = H.T.dot(np.linalg.pinv(X.dot(H.T)).dot(Y))

    def increment(self, X, Y):
        raise NotImplementedError()

    def predict(self, x):
        if self.bias:
            if len(x.shape) == 1:
                x = np.hstack((x, np.ones(1)))
            else:
                x = np.hstack((x, np.ones((x.shape[0], 1))))
        return np.dot(x, self.R)
