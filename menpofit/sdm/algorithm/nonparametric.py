from functools import partial
from menpo.feature import no_op
from menpofit.result import (NonParametricAlgorithmResult,
                             euclidean_bb_normalised_error)

from .base import (BaseSupervisedDescentAlgorithm,
                   compute_non_parametric_delta_x, features_per_image,
                   features_per_patch, update_non_parametric_estimates,
                   print_non_parametric_info)
from menpofit.math import IIRLRegression, IRLRegression, PCRRegression, \
    CCARegression


class NonParametricSDAlgorithm(BaseSupervisedDescentAlgorithm):
    r"""
    """

    def __init__(self):
        super(NonParametricSDAlgorithm, self).__init__()
        self.regressors = []

    def _compute_delta_x(self, gt_shapes, current_shapes):
        return compute_non_parametric_delta_x(gt_shapes, current_shapes)

    def _update_estimates(self, estimated_delta_x, delta_x, gt_x,
                          current_shapes):
        update_non_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                        current_shapes)

    def _compute_training_features(self, images, gt_shapes, current_shapes,
                                   prefix='', verbose=False):
        return features_per_image(images, current_shapes, self.patch_shape,
                                  self.patch_features, prefix=prefix,
                                  verbose=verbose)

    def _compute_test_features(self, image, current_shape):
        return features_per_patch(image, current_shape,
                                  self.patch_shape, self.patch_features)

    def run(self, image, initial_shape, gt_shape=None, **kwargs):
        # set current shape and initialize list of shapes
        current_shape = initial_shape
        shapes = [initial_shape]

        # Cascaded Regression loop
        for r in self.regressors:
            # compute regression features
            features = self._compute_test_features(image, current_shape)

            # solve for increments on the shape vector
            dx = r.predict(features)

            # update current shape
            current_shape = current_shape.from_vector(
                current_shape.as_vector() + dx)
            shapes.append(current_shape)

        # return algorithm result
        return NonParametricAlgorithmResult(image, shapes,
                                            gt_shape=gt_shape)

    def _print_regression_info(self, template_shape, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        print_non_parametric_info(template_shape, gt_shapes, n_perturbations,
                                  delta_x, estimated_delta_x, level_index,
                                  self._compute_error, prefix=prefix)


# TODO: document me!
class NonParametricNewton(NonParametricSDAlgorithm):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, alpha=0, bias=True):
        super(NonParametricNewton, self).__init__()

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


# TODO: document me!
class NonParametricGaussNewton(NonParametricSDAlgorithm):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, alpha=0, bias=True, alpha2=0):
        super(NonParametricGaussNewton, self).__init__()

        self._regressor_cls = partial(IIRLRegression, alpha=alpha, bias=bias,
                                      alpha2=alpha2)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


class NonParametricPCRRegression(NonParametricSDAlgorithm):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, variance=None, bias=True, normalise_x=True):
        super(NonParametricPCRRegression, self).__init__()

        self._regressor_cls = partial(PCRRegression, variance=variance,
                                      bias=bias, normalise_x=normalise_x)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


class NonParametricCCARegression(NonParametricSDAlgorithm):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, variance=None, bias=True):
        super(NonParametricCCARegression, self).__init__()

        self._regressor_cls = partial(CCARegression, variance=variance,
                                      bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps
