import numpy as np
from functools import partial
from menpo.feature import no_op
from menpofit.result import (NonParametricAlgorithmResult,
                             euclidean_bb_normalised_error)

from .base import (BaseSupervisedDescentAlgorithm,
                   compute_parametric_delta_x, features_per_patch,
                   update_parametric_estimates, print_parametric_info)
from menpo.model import PCAModel
from menpo.visualize import print_dynamic
from menpofit.math import IIRLRegression, IRLRegression, OPPRegression
from menpofit.modelinstance import OrthoPDM
from menpofit.visualize import print_progress


class FullyParametricSDAlgorithm(BaseSupervisedDescentAlgorithm):
    r"""
    """

    def __init__(self, shape_model_cls=OrthoPDM, appearance_model_cls=PCAModel):
        super(FullyParametricSDAlgorithm, self).__init__()
        self.regressors = []
        self.shape_model_cls = shape_model_cls
        self.appearance_model_cls = appearance_model_cls
        self.appearance_model = None

    def _compute_delta_x(self, gt_shapes, current_shapes):
        # This is called first - so train shape model here
        self.shape_model = self.shape_model_cls(gt_shapes)

        return compute_parametric_delta_x(gt_shapes, current_shapes,
                                          self.shape_model)

    def _update_estimates(self, estimated_delta_x, delta_x, gt_x,
                          current_shapes):
        update_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                    current_shapes, self.shape_model)

    def _compute_training_features(self, images, gt_shapes, current_shapes,
                                   prefix='', verbose=False):
        if self.appearance_model is None:
            wrap = partial(print_progress,
                           prefix='{}Extracting ground truth patches'.format(prefix),
                           end_with_newline=not prefix, verbose=verbose)
            n_images = len(images)
            # Extract patches from ground truth
            gt_patches = [features_per_patch(im, gt_s, self.patch_shape,
                                             self.patch_features)
                          for gt_s, im in wrap(zip(gt_shapes, images))]
            # Calculate appearance model from extracted gt patches
            gt_patches = np.array(gt_patches).reshape([n_images, -1])
            if verbose:
                print_dynamic('{}Building Appearance Model'.format(prefix))
            self.appearance_model = self.appearance_model_cls(gt_patches)

        wrap = partial(print_progress,
                       prefix='{}Extracting patches'.format(prefix),
                       end_with_newline=not prefix, verbose=verbose)

        features = []
        for im, shapes in wrap(zip(images, current_shapes)):
            for s in shapes:
                param_feature = self._compute_test_features(im, s)
                features.append(param_feature)

        return np.vstack(features)

    def _compute_parametric_features(self, patch):
        raise NotImplementedError()

    def _compute_test_features(self, image, current_shape):
        patch_feature = features_per_patch(
            image, current_shape, self.patch_shape, self.patch_features)
        return self._compute_parametric_features(patch_feature)

    def run(self, image, initial_shape, gt_shape=None, **kwargs):
        # set current shape and initialize list of shapes
        self.shape_model.set_target(initial_shape)
        current_shape = initial_shape.from_vector(
            self.shape_model.target.as_vector().copy())
        shapes = [current_shape]

        # Cascaded Regression loop
        for r in self.regressors:
            # compute regression features
            features = self._compute_test_features(image, current_shape)

            # solve for increments on the shape vector
            dx = r.predict(features).ravel()

            # update current shape
            p = self.shape_model.as_vector() + dx
            self.shape_model.from_vector_inplace(p)
            current_shape = current_shape.from_vector(
                self.shape_model.target.as_vector().copy())
            shapes.append(current_shape)

        # return algorithm result
        return NonParametricAlgorithmResult(image, shapes,
                                            gt_shape=gt_shape)

    def _print_regression_info(self, _, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        print_parametric_info(self.shape_model, gt_shapes, n_perturbations,
                              delta_x, estimated_delta_x, level_index,
                              self._compute_error, prefix=prefix)


class ParametricAppearanceProjectOut(FullyParametricSDAlgorithm):

    def _compute_parametric_features(self, patch):
        return self.appearance_model.project_out_vector(patch.ravel())


class FullyParametricProjectOutNewton(ParametricAppearanceProjectOut):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 appearance_model_cls=PCAModel,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, alpha=0, bias=True):
        super(FullyParametricProjectOutNewton, self).__init__(
            shape_model_cls=shape_model_cls,
            appearance_model_cls=appearance_model_cls)

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


# TODO: document me!
class FullyParametricProjectOutGaussNewton(ParametricAppearanceProjectOut):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 appearance_model_cls=PCAModel,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, alpha=0, bias=True, alpha2=0):
        super(FullyParametricProjectOutGaussNewton, self).__init__(
            shape_model_cls=shape_model_cls,
            appearance_model_cls=appearance_model_cls)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha, bias=bias,
                                      alpha2=alpha2)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


class FullyParametricProjectOutOPP(ParametricAppearanceProjectOut):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 appearance_model_cls=PCAModel,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, bias=True):
        super(FullyParametricProjectOutOPP, self).__init__(
            shape_model_cls=shape_model_cls,
            appearance_model_cls=appearance_model_cls)

        self._regressor_cls = partial(OPPRegression, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps
