from __future__ import division
from functools import partial
import numpy as np
from menpo.visualize import print_dynamic
from menpofit.visualize import print_progress


# TODO: document me!
class BaseSupervisedDescentAlgorithm(object):
    r"""
    """

    def train(self, images, gt_shapes, current_shapes, prefix='',
              verbose=False):
        return self._train(images, gt_shapes, current_shapes, increment=False,
                           prefix=prefix, verbose=verbose)

    def increment(self, images, gt_shapes, current_shapes, prefix='',
                  verbose=False):
        return self._train(images, gt_shapes, current_shapes, increment=True,
                           prefix=prefix, verbose=verbose)

    def _train(self, images, gt_shapes, current_shapes, increment=False,
               prefix='', verbose=False):

        if not increment:
            # Reset the regressors
            self.regressors = []
        elif increment and not (hasattr(self, 'regressors') and self.regressors):
            raise ValueError('Algorithm must be trained before it can be '
                             'incremented.')

        n_perturbations = len(current_shapes[0])
        template_shape = gt_shapes[0]

        # obtain delta_x and gt_x
        delta_x, gt_x = self._compute_delta_x(gt_shapes, current_shapes)

        # Cascaded Regression loop
        for k in range(self.n_iterations):
            # generate regression data
            features_prefix = '{}(Iteration {}) - '.format(prefix, k)
            features = self._compute_training_features(images, gt_shapes,
                                                       current_shapes,
                                                       prefix=features_prefix,
                                                       verbose=verbose)

            if verbose:
                print_dynamic('{}(Iteration {}) - Performing regression'.format(
                    prefix, k))

            if not increment:
                r = self._regressor_cls()
                r.train(features, delta_x)
                self.regressors.append(r)
            else:
                self.regressors[k].increment(features, delta_x)

            # Estimate delta_points
            estimated_delta_x = self.regressors[k].predict(features)
            if verbose:
                self._print_regression_info(template_shape, gt_shapes,
                                            n_perturbations, delta_x,
                                            estimated_delta_x, k,
                                            prefix=prefix)

            self._update_estimates(estimated_delta_x, delta_x, gt_x,
                                   current_shapes)

        return current_shapes

    def _compute_delta_x(self, gt_shapes, current_shapes):
        raise NotImplementedError()

    def _update_estimates(self, estimated_delta_x, delta_x, gt_x,
                          current_shapes):
        raise NotImplementedError()

    def _compute_training_features(self, images, gt_shapes, current_shapes,
                                   prefix='', verbose=False):
        raise NotImplementedError()

    def _compute_test_features(self, image, current_shape):
        raise NotImplementedError()

    def run(self, image, initial_shape, gt_shape=None, **kwargs):
        raise NotImplementedError()

    def _print_regression_info(self, template_shape, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        raise NotImplementedError()



# TODO: document me!
def features_per_patch(image, shape, patch_shape, features_callable):
    """r
    """
    patches = image.extract_patches(shape, patch_shape=patch_shape,
                                    as_single_array=True)

    patch_features = [features_callable(p[0]).ravel() for p in patches]
    return np.hstack(patch_features)


# TODO: document me!
def features_per_shapes(image, shapes, patch_shape, features_callable):
    """r
    """
    patch_features = [features_per_patch(image, s, patch_shape,
                                         features_callable)
                      for s in shapes]

    return np.vstack(patch_features)


# TODO: document me!
def features_per_image(images, shapes, patch_shape, features_callable,
                       prefix='', verbose=False):
    """r
    """
    wrap = partial(print_progress,
                   prefix='{}Extracting patches'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)

    patch_features = [features_per_shapes(i, shapes[j], patch_shape,
                                          features_callable)
                      for j, i in enumerate(wrap(images))]
    return np.vstack(patch_features)


def compute_non_parametric_delta_x(gt_shapes, current_shapes):
    r"""
    """
    n_x = gt_shapes[0].n_parameters
    n_gt_shapes = len(gt_shapes)
    n_current_shapes = len(current_shapes[0])

    # initialize current, ground truth and delta parameters
    gt_x = np.empty((n_gt_shapes * n_current_shapes, n_x))
    delta_x = np.empty((n_gt_shapes * n_current_shapes, n_x))

    # obtain ground truth points and compute delta points
    k = 0
    for gt_s, shapes in zip(gt_shapes, current_shapes):
        c_gt_s = gt_s.as_vector()
        for s in shapes:
            # compute ground truth shape vector
            gt_x[k] = c_gt_s
            # compute delta shape vector
            delta_x[k] = c_gt_s - s.as_vector()
            k += 1

    return delta_x, gt_x


def update_non_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                    current_shapes):
    j = 0
    for shapes in current_shapes:
        for s in shapes:
            # update current x
            current_x = s.as_vector() + estimated_delta_x[j]
            # update current shape inplace
            s.from_vector_inplace(current_x)
            # update delta_x
            delta_x[j] = gt_x[j] - current_x
            # increase index
            j += 1


def print_non_parametric_info(template_shape, gt_shapes, n_perturbations,
                              delta_x, estimated_delta_x, level_index,
                              compute_error_f, prefix=''):
    print_dynamic('{}(Iteration {}) - Calculating errors'.format(
        prefix, level_index))
    errors = []
    for j, (dx, edx) in enumerate(zip(delta_x, estimated_delta_x)):
        s1 = template_shape.from_vector(dx)
        s2 = template_shape.from_vector(edx)
        gt_s = gt_shapes[np.floor_divide(j, n_perturbations)]
        errors.append(compute_error_f(s1, s2, gt_s))
    mean = np.mean(errors)
    std = np.std(errors)
    median = np.median(errors)
    print_dynamic('{}(Iteration {}) - Training error -> '
                  'mean: {:.4f}, std: {:.4f}, median: {:.4f}.\n'.
                  format(prefix, level_index, mean, std, median))


def print_parametric_info(model, gt_shapes, n_perturbations,
                          delta_x, estimated_delta_x, level_index,
                          compute_error_f, prefix=''):
    print_dynamic('{}(Iteration {}) - Calculating errors'.format(
        prefix, level_index))
    errors = []
    for j, (dx, edx) in enumerate(zip(delta_x, estimated_delta_x)):
        model.from_vector_inplace(dx)
        s1 = model.target
        model.from_vector_inplace(edx)
        s2 = model.target

        gt_s = gt_shapes[np.floor_divide(j, n_perturbations)]
        errors.append(compute_error_f(s1, s2, gt_s))
    mean = np.mean(errors)
    std = np.std(errors)
    median = np.median(errors)
    print_dynamic('{}(Iteration {}) - Training error -> '
                  'mean: {:.4f}, std: {:.4f}, median: {:.4f}.\n'.
                  format(prefix, level_index, mean, std, median))


def compute_parametric_delta_x(gt_shapes, current_shapes, model):
    # initialize current and delta parameters arrays
    n_samples = len(gt_shapes) * len(current_shapes[0])
    gt_params = np.empty((n_samples, model.n_parameters))
    delta_params = np.empty_like(gt_params)

    k = 0
    for gt_s, c_s in zip(gt_shapes, current_shapes):
        # Compute and cache ground truth parameters
        c_gt_params = model.set_target(gt_s).as_vector()
        for s in c_s:
            gt_params[k] = c_gt_params

            current_params = model.set_target(s).as_vector()
            delta_params[k] = c_gt_params - current_params

            k += 1

    return delta_params, gt_params


def update_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                current_shapes, model):
    j = 0
    for shapes in current_shapes:
        for s in shapes:
            # Estimate parameters
            edx = estimated_delta_x[j]
            # Current parameters
            cx = model.set_target(s).as_vector() + edx
            model.from_vector_inplace(cx)

            # Update current shape inplace
            s.from_vector_inplace(model.target.as_vector())

            delta_x[j] = gt_x[j] - cx
            j += 1
