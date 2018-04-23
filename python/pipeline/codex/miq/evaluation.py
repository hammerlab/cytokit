from __future__ import print_function
"""
https://github.com/google/microscopeimagequality/blob/main/microscopeimagequality/evaluation.py
"""

import collections
import csv
import logging
import os

# import PIL.Image
# import PIL.ImageDraw
# import matplotlib.pyplot
import numpy
import scipy.misc
import scipy.stats
# import skimage.io
import tensorflow
import tensorflow.contrib.slim
import tensorflow.python.ops

# import microscopeimagequality.miq
import codex.miq.miq

_IMAGE_ANNOTATION_MAGNIFICATION_PERCENT = 800
CERTAINTY_NAMES = ['mean', 'max', 'aggregate', 'weighted']
CERTAINTY_TYPES = {i: CERTAINTY_NAMES[i] for i in range(len(CERTAINTY_NAMES))}
BORDER_SIZE = 8

CLASS_ANNOTATION_COLORMAP = 'hsv'

METHOD_AVERAGE = 'average'
METHOD_PRODUCT = 'product'


class WholeImagePrediction(
    collections.namedtuple('WholeImagePrediction', ['predictions', 'certainties', 'probabilities'])):
    """
    Prediction for a whole image.

    Properties:
        predictions: The integer index representing the class with highest average probability. For 11
          focus classes, this is a value in [0, 11), where lower denotes more in-focus.
        certainties: A dictionary mapping prediction certainty type to float certainty values.
        probabilities: 1D numpy float array of the class probabilities.
    """


class ModelAndMetrics(collections.namedtuple('ModelAndMetrics', ['logits', 'labels', 'probabilities', 'predictions'])):
    """
    Object for model and metrics tensors.

    Properties:
        logits: Tensor of logits of size [batch_size x num_classes].
        labels: Tensor of labels of size [batch_size].
        probabilities: Tensor of probabilities of size [batch_size x num_classes].
        predictions: Tensor of predictions of size [batch_size].
    """


# def annotate_patch(image, prediction, label):
#     """Annotates image with classification result. Use with tf.py_func().
#
#   Args:
#     image: Numpy array of shape [1, image_width, image_width, 1].
#     prediction: Integer representing predicted class.
#     label: Integer representing actual class.
#   Returns:
#     Annotated image as a numpy array of shape [1, new_width, new_width, 1].
#   """
#     if prediction == label:
#         text_label = 'actual/predicted: %g' % label
#     else:
#         text_label = 'ERROR: actual: %g, predicted: %g' % (label, prediction)
#
#     # Enlarge the image so the text is legible.
#     resized_image = scipy.misc.imresize(
#         numpy.squeeze(image),
#         size=float(_IMAGE_ANNOTATION_MAGNIFICATION_PERCENT) / 100.0,
#         interp='nearest')
#
#     # Use PIL image to introduce a text label, then convert back to numpy array.
#     pil_image = PIL.Image.fromarray(resized_image)
#     draw = PIL.ImageDraw.Draw(pil_image)
#     draw.text((0, 0), text_label, 255)
#     annotated_image = numpy.asarray(pil_image, dtype=image.dtype)
#
#     # Expand from [new_width, new_width] shape to 4D shape required by TensorFlow.
#     annotated_image_expanded = numpy.expand_dims(
#         numpy.expand_dims(
#             annotated_image, axis=0), axis=3)
#
#     return annotated_image_expanded


# def annotate_classification_errors(images, predictions, labels, probabilities,
#                                    image_height, image_width):
#     """Annotate images with classification errors for TensorBoard viewing.
#
#   Args:
#     images: Tensor of images, of size [batch_size x image_width x
#       image_width x 1].
#     predictions: Tensor of predictions.
#     labels: Tensor of labels.
#     probabilities: Tensor of probabilities.
#     image_height: Integer, the image height.
#     image_width: Integer, the image width.
#
#   Returns:
#     Tuple of image and summary Tensors.
#   """
#
#     for i in range(images.get_shape().as_list()[0]):
#         label = tensorflow.squeeze(tensorflow.strided_slice(labels, [i], [i + 1]))
#         prediction = tensorflow.squeeze(tensorflow.strided_slice(predictions, [i], [i + 1]))
#         patch = tensorflow.strided_slice(images, [i, 0, 0, 0], [
#             i + 1, images.get_shape().as_list()[1], images.get_shape().as_list()[2],
#             images.get_shape().as_list()[3]
#         ])
#
#         patch_annotated = tensorflow.py_func(annotate_patch, [patch, prediction, label],
#                                              [patch.dtype])[0]
#
#         tensorflow.summary.image('Patch_%02d' % i, patch_annotated)
#     image = tensorflow.py_func(visualize_image_predictions,
#                                [images, probabilities, labels, image_height, image_width],
#                                [tensorflow.uint8])[0]
#     summary = tensorflow.summary.image('Annotated_Image_', image)
#     return image, summary


# def visualize_image_predictions(patches,
#                                 probabilities,
#                                 labels,
#                                 image_height,
#                                 image_width,
#                                 show_plot=False,
#                                 output_path=None,
#                                 apply_gamma=False):
#     """Stitch patches into image with color annotations. Use with tf.py_func().
#
#   A colored border will be added to each patch based on the predicted class.
#   Also, colored bars at the top and bottom will indicate the entire image true
#   label and prediction (the most probable class after averaging the patch
#   probabilities).
#   Args:
#     patches: Numpy array of patches of shape (num_patches, width, width, 1).
#     probabilities: Numpy array of shape (num_patches, num_classes), the
#       probabilities predicted by the model for each class.
#     labels: Integer numpy array of shape (num_patches) indicating true class
#       show_plot. The true class must be the same for all patches.
#     image_height: Integer, the image height.
#     image_width: Integer, the image width.
#     show_plot: Boolean, whether to show plot (use this in Colab).
#     output_path: String, path to save annotated image.
#     apply_gamma: Boolean, whether to apply gamma for visualization purposes.
#
#   Returns:
#     RGB image as numpy array of shape (1, image_width, image_width, 3).
#   """
#     assert len(patches.shape) == 4
#     assert patches.shape[0] == probabilities.shape[0]
#     assert numpy.all(labels == labels[0])
#
#     image_rgb = get_rgb_image(
#         max(1.0 / 65535, numpy.max(patches)),
#         patches,
#         probabilities,
#         labels, (image_height, image_width),
#         apply_gamma=apply_gamma)
#
#     # Plot it.
#     if show_plot:
#         matplotlib.pyplot.figure(figsize=(6, 6))
#         matplotlib.pyplot.imshow(image_rgb, interpolation='nearest', cmap='gray')
#         matplotlib.pyplot.grid('off')
#
#     # Save it.
#     if output_path is not None:
#         skimage.io.imsave(output_path, image_rgb)
#
#     # Expand from to 4D shape required by TensorFlow.
#     return numpy.expand_dims(image_rgb, 0)


# def _get_class_rgb(num_classes, predicted_class):
#     """Map from class to RGB value for a specific colormap.
#
#   Args:
#     num_classes: Integer, the total number of classes.
#     predicted_class: Integer, the predicted class, in [0, num_classes).
#
#   Returns:
#     Tuple of 3 floats in [0.0, 1.0] representing an RGB color.
#
#   Raises:
#     ValueError: If predicted class is not in [0, num_classes).
#   """
#     if not 0 <= predicted_class < num_classes:
#         raise ValueError('Predicted class %d must be in [0, %d).' %
#                          (predicted_class, num_classes))
#     # Map [0, num_classes) to [0, 255)
#     colormap_index = int(predicted_class * 255.0 / num_classes)
#     # Return just the RGB values of the colormap.
#     return matplotlib.pyplot.cm.get_cmap(CLASS_ANNOTATION_COLORMAP)(colormap_index)[0:3]


def get_certainty(probabilities):
    """Get a measure of certainty in [0.0, 1.0] given the class probabilities.

  Args:
    probabilities: A float numpy array of size num_classes, a probability
      distribution.

  Returns:
    A float in the range [0.0, 1.0] representing the certainty of the
    distribution.
  """
    sum_prob = numpy.sum(probabilities)
    num_classes = probabilities.shape[0]
    if sum_prob > 0:
        normalized_probabilities = probabilities / sum_prob

        certainty_proxy = 1.0 - scipy.stats.entropy(
            normalized_probabilities) / numpy.log(num_classes)

    else:
        certainty_proxy = 0.0
    assert certainty_proxy - 1 < 1e-6, 'certainty: ' ' %g' % certainty_proxy
    assert certainty_proxy > -1e-6, 'certainty:' ' %g' % certainty_proxy
    certainty_proxy = numpy.clip(certainty_proxy, 0.0, 1.0)
    return certainty_proxy


# def get_rgb_image(max_value,
#                   patches,
#                   probabilities,
#                   labels,
#                   image_shape,
#                   apply_gamma=False):
#     """Add colored borders to patches based on predictions and get whole image.
#
#   Args:
#     max_value: The max pixel value of the image, to which all annotations will
#       be scaled to.
#     patches: Numpy array of patches of shape (num_patches, width, width, 1).
#     probabilities: Numpy array of shape (num_patches, num_classes), the
#       probabilities predicted by the model for each class.
#     labels: Integer numpy array of shape (num_patches) indicating true class.
#       The true class must be the same for all patches. A value of '-1' denotes
#       that no true label exists.
#     image_shape: Tuple of integers, the height and width of assembled image.
#     apply_gamma: Boolean, whether to apply a gamma transform for visualization.
#
#   Returns:
#     The whole-image (assembled patches) a 3D numpy array with dtype uint8
#     representing a 2D RGB image, with annotations for patch and whole-image
#     predictions.
#   """
#     assert patches.shape[3] == 1
#     num_classes = probabilities.shape[1]
#
#     patches_rgb = numpy.zeros(
#         (patches.shape[0], patches.shape[1], patches.shape[2], 3))
#
#     for i in range(patches.shape[0]):
#         patch = patches[i, :, :, :]
#
#         prediction = numpy.argmax(probabilities[i, :])
#
#         certainty_proxy = get_certainty(probabilities[i, :])
#
#         # The brightness of the annotation should map from no certainty (random
#         # probability) to 100% certainty, to the range [0 - 1.0].
#         class_rgb = _get_class_rgb(num_classes, prediction)
#
#         class_rgb_with_certainty = [
#             numpy.float(max_value * certainty_proxy * c) for c in class_rgb
#         ]
#         patches_rgb[i, :, :, :] = numpy.concatenate(
#             (_set_border_pixels(patch, class_rgb_with_certainty[0]),
#              _set_border_pixels(patch, class_rgb_with_certainty[1]),
#              _set_border_pixels(patch, class_rgb_with_certainty[2])),
#             axis=2)
#
#     image_rgb = _patches_to_image(patches_rgb, image_shape)
#     predicted_color = _get_class_rgb(
#         num_classes, aggregate_prediction_from_probabilities(probabilities)[0])
#
#     if labels[0] == -1:
#         actual_color = None
#     else:
#         actual_color = _get_class_rgb(num_classes, labels[0])
#     image_rgb = _add_rgb_annotation(image_rgb, predicted_color, actual_color,
#                                     max_value)
#
#     if apply_gamma:
#         image_rgb = apply_image_gamma(image_rgb)
#     image_rgb = (255 * image_rgb / numpy.max(image_rgb)).astype(numpy.uint8)
#     return image_rgb


def certainties_from_probabilities(probabilities):
    """Get certainty for each set of predicted probabilities.

  Certainty is a number from 0.0 to 1.0, with 1.0 indicating a prediction with
  100% probability in one class, and 0.0 indicating a uniform probability over
  all classes.

  Args:
    probabilities: Numpy array of marginal probabilities, shape
     (batch_size, num_classes).

  Returns:
    Numpy array of certainties, of shape (batch_size).
  """
    certainties = numpy.zeros(probabilities.shape[0])
    for i in range(probabilities.shape[0]):
        certainties[i] = get_certainty(probabilities[i, :])
    return certainties


def aggregate_prediction_from_probabilities(probabilities,
                                            aggregation_method=METHOD_AVERAGE):
    """Determine the whole-image class prediction from patch probabilities.

  Args:
    probabilities: Numpy array of marginal probabilities, shape
     (batch_size, num_classes).
    aggregation_method: String, the method of aggregating the patch
      probabilities.

  Returns:
    A WholeImagePrediction object.

  Raises:
    ValueError: If the aggregation method is not valid.
  """
    certainties = certainties_from_probabilities(probabilities)

    certainty_dict = {
        'mean': numpy.round(numpy.mean(certainties), 3),
        'max': numpy.round(numpy.max(certainties), 3)
    }

    weights = certainties
    weights = None if numpy.sum(weights) == 0 else weights

    if aggregation_method == METHOD_AVERAGE:
        probabilities_aggregated = numpy.average(probabilities, 0, weights=weights)
    elif aggregation_method == METHOD_PRODUCT:
        # For i denoting index within batch and c the class:
        #   Q_c = product_over_i(p_c(i))
        # probabilities_aggregated = Q_c / sum_over_c(Q_c)
        # The following computes this using logs for numerical stability.
        sum_log_probabilities = numpy.sum(numpy.log(probabilities), 0)
        probabilities_aggregated = numpy.exp(
            sum_log_probabilities - scipy.misc.logsumexp(sum_log_probabilities))
    else:
        raise ValueError('Invalid aggregation method %s.' % aggregation_method)
    predicted_class = numpy.argmax(probabilities_aggregated)
    certainty_dict['aggregate'] = numpy.round(
        get_certainty(probabilities_aggregated), 3)
    certainty_dict['weighted'] = numpy.round(
        numpy.average(
            certainties, 0, weights=weights), 3)

    assert sorted(CERTAINTY_TYPES.values()) == sorted(certainty_dict.keys())

    return WholeImagePrediction(predicted_class, certainty_dict,
                                probabilities_aggregated)


def _add_rgb_annotation(image, predicted_color, actual_color, max_value):
    """Adds color actual/predicted annotations to top and bottom of image.

  Args:
    image: Numpy array representing a 2D RGB image to annotate.
    predicted_color: Tuple of length 3 of RGB float values in [0.0, 1.0].
    actual_color: Tuple of length 3 of RGB float values in [0.0, 1.0]. None if
      no
      actual class annotation should be applied.
    max_value: The value which an RGB value of 1.0 should be mapped to.

  Returns:
    The original image, same size and type, but with colored annotations.
  """
    assert len(image.shape) == 3
    for i in range(3):
        if actual_color:
            image[0:BORDER_SIZE, :, i] = actual_color[i] * max_value
        image[-1 * BORDER_SIZE:, :, i] = predicted_color[i] * max_value
    return image


def _patches_to_image(patches, image_shape):
    """Reshapes a numpy array of patches to a single image.

  Args:
    patches: Numpy array of shape (num_patches, patch_width, patch_width, 1).
    image_shape: Tuple of integers, the height and width of assembled image.

  Returns:
    The whole assembled image, shape (image_shape[0], image_shape[1], 1).

  Raises:
     ValueError: If the input array dimensions are incorrect.
  """
    if len(patches.shape) != 4:
        raise ValueError('Input array has shape %s but must be 4D.' %
                         str(patches.shape))
    num_patches = patches.shape[0]
    patch_width = patches.shape[1]
    num_rows = image_shape[0] // patch_width
    num_cols = image_shape[1] // patch_width

    if num_rows * num_cols != num_patches:
        raise ValueError('image_shape %s not valid for %d %dx%d patches.' %
                         (str(image_shape), num_patches, patch_width, patch_width))

    image = numpy.zeros([num_rows * patch_width, num_cols * patch_width, patches.shape[3]], dtype=patches.dtype)

    index = 0
    for i in range(0, num_rows * patch_width, patch_width):
        for j in range(0, num_cols * patch_width, patch_width):
            image[i:i + patch_width, j:j + patch_width, :] = patches[index, :, :, :]
            index += 1

    return image


def _set_border_pixels(patch, value, border_size=2):
    """Sets border pixels in 2D grayscale image.

  Args:
    patch: Numpy array of shape (patch_width, patch_width, 1).
    value: Value to set the border pixels to.
    border_size: Integer, the width of the border to add, in pixels.

  Returns:
    A numpy array of same size as 'patch', with the border pixels modified.
  """
    assert len(patch.shape) == 3
    assert patch.shape[2] == 1
    return numpy.expand_dims(
        numpy.pad(patch[border_size:-border_size, border_size:-border_size, 0],
                  border_size,
                  'constant',
                  constant_values=value),
        2)


def apply_image_gamma(original_image, gamma=2.2):
    """Applies image gamma for easier viewing.

  Args:
    original_image: Numpy array of any shape.
    gamma: Float, the gamma value to modify each pixel with.

  Returns:
    A numpy array of same shape and type as the input image, but with a gamma
    transform applied independently at each pixel.
  """
    image = numpy.copy(original_image).astype(numpy.float32)
    max_value = numpy.max(image)
    image /= max_value
    image = numpy.power(image, 1 / gamma)
    image *= max_value
    return image.astype(original_image.dtype)


def get_aggregated_prediction(probabilities, labels, batch_size):
    """Aggregates all probabilities in a batch into a single prediction.

    Args:
    probabilities: Tensor of probabilities of size [batch_size x num_classes].
    labels: Tensor of labels of size [batch_size].
    batch_size: Integer representing number of samples per batch.
    Returns:
    The prediction is the class with highest average probability across the
    batch, as a single-element Tensor and the true label (single-element
    Tensor). All elements in `labels` must be indentical.
    """

    # We aggregate the probabilities by using a weighted average.
    def aggregate_prediction(probs):
        return aggregate_prediction_from_probabilities(probs).predictions.astype(
            numpy.int64)

    prediction = tensorflow.py_func(aggregate_prediction, [probabilities], tensorflow.int64)

    # Check that all batch labels are the same class.
    max_label = tensorflow.reduce_max(labels)

    with tensorflow.control_dependencies([tensorflow.assert_equal(
            tensorflow.multiply(
                max_label, tensorflow.constant(
                    batch_size, dtype=max_label.dtype)),
            tensorflow.reduce_sum(labels),
            name='check_all_batch_labels_same')]):
        label = tensorflow.reduce_mean(labels)

        # Since the Tensor shape cannot be inferred by py_func() manually annotate it.
        prediction.set_shape(label.get_shape())

    return prediction, label



def get_model_and_metrics(images,
                          num_classes,
                          one_hot_labels,
                          is_training,
                          model_id=0):
    """Get the model and metrics.

  Args:
    images: A `Tensor` of size [batch_size, patch_width, patch_width, 1]
    num_classes: Integer representing number of classes.
    one_hot_labels: A `Tensor` of size [batch_size, num_classes], where
      each row has a single element set to one and the rest set to zeros.
    is_training: Boolean, whether the model is training.
    model_id: Integer, model ID.

  Returns:
    A ModelAndMetrics object.
  """
    # Define the model:
    logits = codex.miq.miq.miq_model(
        images,
        num_classes=num_classes,
        is_training=is_training,
        model_id=model_id)

    # Define the metrics:
    # If there exists no label for the ith row, then one_hot_labels[:,i] will all
    # be zeros. In this case, labels[i] should be -1. Otherwise, labels[i]
    # reflects the true class.
    label_exists = tensorflow.equal(tensorflow.reduce_sum(one_hot_labels, 1), 1)
    label_for_unlabeled_data = tensorflow.multiply(
        tensorflow.constant(-1, dtype=tensorflow.int64),
        tensorflow.ones([tensorflow.shape(one_hot_labels)[0]], dtype=tensorflow.int64))
    labels = tensorflow.where(label_exists,
                              tensorflow.argmax(one_hot_labels, 1), label_for_unlabeled_data)
    probabilities = tensorflow.nn.softmax(logits)
    predictions = tensorflow.argmax(logits, 1)

    return ModelAndMetrics(logits, labels, probabilities, predictions)

