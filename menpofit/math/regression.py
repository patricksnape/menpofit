import numpy as np


from menpo.math import pca
from menpo.math.decomposition import cca


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
        self.W = np.linalg.solve(XX, X.T.dot(Y))

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
        np.fill_diagonal(H, self.alpha2 + np.diag(H))
        self.W = np.linalg.solve(H, J).T
        # self.W = np.linalg.pinv(J)

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
        return np.dot(x, self.W)


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
    def __init__(self, variance=None, bias=True,eps=1e-10):
        self.variance = variance
        self.bias = bias
        self.R = None
        self.V = None
        self.eps = eps

    def train(self, X, Y):
        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Reduce variance
        U, s, self.V = np.linalg.svd(X, full_matrices=False)
        if self.variance is not None:
            variation = np.cumsum(s) / np.sum(s)
            # Inverted for easier parameter semantics
            k = np.sum(variation < self.variance)
            U = U[:, :k]
            self.V = self.V[:k, :]
            s = s[:k]
        S = np.diag(s)

        # Perform PCR
        self.R = self.V.T.dot(np.linalg.inv(S)).dot(U.T).dot(Y)

    def increment(self, X, Y):
        raise NotImplementedError()

    def predict(self, x):
        if self.bias:
            if len(x.shape) == 1:
                x = np.hstack((x, np.ones(1)))
            else:
                x = np.hstack((x, np.ones((x.shape[0], 1))))
        x = np.dot(np.dot(x, self.V.T), self.V)
        return np.dot(x, self.R)


class OptimalLinearRegression(object):
    r"""
    Multivariate Linear Regression using optimal reconstructions

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
        k = U.shape[0]
        if self.variance is not None:
            variation = np.cumsum(l) / np.sum(l)
            # Inverted for easier parameter semantics
            k = np.sum(variation < self.variance)
            U = U[:k, :]

        # Whitened components
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


class CCARegression(object):
    r"""
    Canonical Correlation Analysis Regression
    """
    def __init__(self, alpha=0, bias=True):
        self.alpha = alpha
        self.bias = bias
        self.R = None
        self.Wx = None
        self.mx = None
        self.my = None
        self.C = None

    def train(self, X, Y):
        # perform cca
        self.Wx, Wy, self.mx, self.my, self.C = cca(X, Y)

        # compute canonical variable
        Zx = (X - self.mx).dot(self.Wx)
        Zy = (Y - self.my).dot(Wy)

        if self.bias:
            # add bias
            Zx = np.hstack((Zx, np.ones((Zx.shape[0], 1))))

        # regularized linear least squares
        ZxZx = Zx.T.dot(Zx)
        if self.alpha:
            np.fill_diagonal(ZxZx, self.alpha + np.diag(ZxZx))
        self.R = np.linalg.solve(ZxZx, Zx.T.dot(Zy))

        # post multiply by the pseudo-inverse of Wy
        self.R = self.R.dot(np.linalg.pinv(Wy))

    def predict(self, x):
        zx = (x - self.mx).dot(self.Wx)
        if self.bias:
            if len(x.shape) == 1:
                zx = np.hstack((zx, np.ones(1)))
            else:
                zx = np.hstack((zx, np.ones((x.shape[0], 1))))
        return self.my + np.dot(zx, self.R)


class OPPRegression(object):
    r"""
    Multivariate Linear Regression using Orthogonal Procrustes Problem
    reconstructions.

    Parameters
    ----------
    X : numpy.array
        The regression features used to create the coefficient matrix.
    T : numpy.array
        The shapes differential that denote the dependent variable.

    Raises
    ------
    ValueError
        variance must be set to a number between 0 and 1
    """
    def __init__(self, bias=True, eps=1e-10, whiten=False):
        self.bias = bias
        self.R = None
        self.eps = eps
        self.whiten = whiten

    def train(self, X, Y):
        if self.bias:
            # add bias
             X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Whiten
        if self.whiten:
            U, l, _ = pca(X, centre=False)
            inv_eig = np.sqrt(np.linalg.inv(np.diag(l)))
            U = inv_eig.dot(U)
            X = X.dot(U.T).dot(U)

        U, _, V = np.linalg.svd(X.T.dot(Y), full_matrices=False)
        # Skinny SVD
        self.R = U.dot(V)

    def increment(self, X, Y):
        raise NotImplementedError()

    def predict(self, x):
        if self.bias:
            if len(x.shape) == 1:
                x = np.hstack((x, np.ones(1)))
            else:
                x = np.hstack((x, np.ones((x.shape[0], 1))))
        return np.dot(x, self.R)
