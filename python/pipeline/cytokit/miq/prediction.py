"""
https://github.com/google/microscopeimagequality/blob/main/microscopeimagequality/prediction.py
"""

import logging
import sys

import numpy
import tensorflow

import cytokit.miq.constants
import cytokit.miq.evaluation

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

_SPLIT_NAME = 'test'

_TFRECORD_FILE_PATTERN = 'data_%s-%05d-of-%05d.tfrecord'

logger = logging.getLogger(__name__)


class ImageQualityClassifier(object):
    """Object for running image quality model inference.

    Attributes:
      graph: TensorFlow graph.
    """

    def __init__(self,
                 model_ckpt,
                 model_patch_side_length,
                 num_classes,
                 graph=None, 
                 session_config=None):
        """Initialize the model from a checkpoint.

        Args:
          model_ckpt: String, path to TensorFlow model checkpoint to load.
          model_patch_side_length: Integer, the side length in pixels of the square
            image passed to the model.
          num_classes: Integer, the number of classes the model predicts.
          graph: TensorFlow graph. If None, one will be created.
          session_config: TensorFlow session configuration.  If None, one will be created
        """
        self._model_patch_side_length = model_patch_side_length
        self._num_classes = num_classes

        if graph is None:
            graph = tensorflow.Graph()
        self.graph = graph

        with self.graph.as_default():
            self._image_placeholder = tensorflow.placeholder(
                tensorflow.float32, shape=[None, None, 1])

            self._probabilities = self._probabilities_from_image(
                self._image_placeholder, model_patch_side_length, num_classes)

            self._sess = tensorflow.Session(config=session_config)
            saver = tensorflow.train.Saver()

            saver.restore(self._sess, model_ckpt)
        logger.debug('Restored image focus prediction model from %s.', model_ckpt)

    def __del__(self):
        self._sess.close()

    def _probabilities_from_image(self, image_placeholder,
                                  model_patch_side_length, num_classes):
        """Get probabilities tensor from input image tensor.

        Args:
          image_placeholder: Float32 tensor, placeholder for input image.
          model_patch_side_length: Integer, the side length in pixels of the square
            image passed to the model.
          num_classes: Integer, the number of classes the model predicts.

        Returns:
          Probabilities tensor, shape [num_classes] representing the predicted
          probabilities for each class.
        """
        labels_fake = tensorflow.zeros([self._num_classes])

        image_path_fake = tensorflow.constant(['unused'])
        tiles, labels, _ = _get_image_tiles_tensor(
            image_placeholder, labels_fake, image_path_fake,
            model_patch_side_length)

        model_metrics = cytokit.miq.evaluation.get_model_and_metrics(
            tiles,
            num_classes=num_classes,
            one_hot_labels=labels,
            is_training=False)

        return model_metrics.probabilities

    def predict(self, image):
        """Run inference on an image.

        Args:
          image: Numpy float array, two-dimensional.

        Returns:
          A evaluation.WholeImagePrediction object.
        """
        feed_dict = {self._image_placeholder: numpy.expand_dims(image, 2)}
        [np_probabilities] = self._sess.run(
            [self._probabilities], feed_dict=feed_dict)

        return cytokit.miq.evaluation.aggregate_prediction_from_probabilities(
            np_probabilities, cytokit.miq.evaluation.METHOD_AVERAGE)

    def get_patch_predictions(self, image):
        """Run inference on each patch in an image, returning each patch score.

        Args:
          image: Numpy float array, of shape (height, width).

        Returns:
          List of tuples, with (upper_left_row, upper_left_col, height, width
          evaluation.WholeImagePrediction) which denote the patch location,
          dimensions and predition result.
        """
        results = []
        w = cytokit.miq.constants.PATCH_SIDE_LENGTH
        for i in range(0, image.shape[0] - w, w):
            for j in range(0, image.shape[1] - w, w):
                results.append((i, j, w, w, self.predict(image[i:i + w, j:j + w])))
        return results

    def get_annotated_prediction(self, image):
        """Run inference to annotate the input image with patch predictions.

        Args:
          image: Numpy float array, two-dimensional.

        Returns:
          RGB image as uint8 numpy array of shape (image_height, image_width, 3),
          representing the upper left crop of the input image, where:
             image_height = floor(image.shape[0] / model_patch_side_length)
             image_width = floor(image.shape[1] / model_patch_side_length)
        """

        feed_dict = {self._image_placeholder: numpy.expand_dims(image, 2)}

        with self.graph.as_default():
            patches = _get_image_tiles_tensor(
                self._image_placeholder,
                tensorflow.constant([0]),
                tensorflow.constant([0]),
                patch_width=self._model_patch_side_length)[0]
            [np_probabilities, np_patches] = self._sess.run(
                [self._probabilities, patches], feed_dict=feed_dict)

        # We use '-1' to denote no true label exists.
        np_labels = -1 * numpy.ones((np_patches.shape[0]))
        return numpy.squeeze(
            cytokit.miq.evaluation.visualize_image_predictions(
                np_patches,
                np_probabilities,
                np_labels,
                image.shape[0],
                image.shape[1],
                show_plot=False,
                output_path=None))


def patch_values_to_mask(values, patch_width):
    """Construct a mask from an array of patch values.

  Args:
    values: A uint16 2D numpy array.
    patch_width: Width in pixels of each patch.

  Returns:
    The  mask, a uint16 numpy array of width patch_width *
    values.shape[0].

  Raises:
    ValueError: If the input values are invalid.
  """
    if values.dtype != numpy.uint16 or len(values.shape) != 2:
        logging.info('dtype: %s shape: %s', values.dtype, values.shape)
        raise ValueError('Input must be a 2D np.uint16 array.')

    patches_per_column = values.shape[0]
    patches_per_row = values.shape[1]

    mask = numpy.zeros(
        (patches_per_column * patch_width, patches_per_row * patch_width),
        dtype=numpy.uint16)

    for i in range(patches_per_column):
        for j in range(patches_per_row):
            ymin = i * patch_width
            xmin = j * patch_width
            mask[ymin:ymin + patch_width, xmin:xmin + patch_width] = values[i, j]

    return mask


def _get_image_tiles_tensor(image, label, image_path, patch_width):
    """Gets patches that tile the input image, starting at upper left.

    Args:
      image: Input image tensor, size [height x width x 1].
      label: Input label tensor, size [num_classes].
      image_path: Input image path tensor, size [1].
      patch_width: Integer representing width of image patch.

    Returns:
      Tensors tiles, size [num_tiles x patch_width x patch_width x 1], labels,
      size [num_tiles x num_classes], and image_paths, size [num_tiles x 1].
    """
    tiles_before_reshape = tensorflow.extract_image_patches(
        tensorflow.expand_dims(image, dim=0), [1, patch_width, patch_width, 1],
        [1, patch_width, patch_width, 1], [1, 1, 1, 1], 'VALID')
    tiles = tensorflow.reshape(tiles_before_reshape, [-1, patch_width, patch_width, 1])

    labels = tensorflow.tile(tensorflow.expand_dims(label, dim=0), [tensorflow.shape(tiles)[0], 1])
    image_paths = tensorflow.tile(
        tensorflow.expand_dims(image_path, dim=0), [tensorflow.shape(tiles)[0], 1])
    return tiles, labels, image_paths

