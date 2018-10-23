"""
https://github.com/google/microscopeimagequality/blob/main/microscopeimagequality/miq.py
"""

import logging
import os
import pkg_resources

import tensorflow
import tensorflow.contrib.slim

import cytokit.miq.constants as constants

DEFAULT_MODEL_DIRECTORY = pkg_resources.resource_filename(__name__, "data")
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIRECTORY + "/" + os.path.basename(constants.REMOTE_MODEL_CHECKPOINT_PATH)

logger = logging.getLogger(__name__)


def add_loss(logits, one_hot_labels, use_rank_loss=False):
    """Add loss function to tf.losses.

  Args:
    logits: Tensor of logits of shape [batch_size, num_classes]
    one_hot_labels: A `Tensor` of size [batch_size, num_classes], where
      each row has a single element set to one and the rest set to zeros.
    use_rank_loss: Boolean, whether to use rank probability score loss instead
      of cross entropy.
  """
    if not use_rank_loss:
        tensorflow.contrib.slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    else:
        rank_loss = ranked_probability_score(
            tensorflow.nn.softmax(logits), one_hot_labels, dim=1)
        tensorflow.losses.add_loss(tensorflow.reduce_mean(rank_loss))


def miq_model(images, num_classes=2, is_training=False, model_id=0):
    """Creates the convolutional model.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:
        logits = miq.Miq(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: the input patches, a tensor of size [batch_size, patch_width,
      patch_width, 1].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    model_id: Integer, model ID.

  Returns:
    the output logits, a tensor of size [batch_size, 11].

  Raises:
    ValueError: If an invalid model ID is specified.
  """
    # logger.debug('Using model_id = %d.', model_id)
    if model_id == 0:
        return model_v0(images, num_classes, is_training)
    elif model_id == 1:
        return model_v1(images, num_classes, is_training)
    else:
        raise ValueError('Unsupported model %d' % model_id)


def model_v1(images, num_classes, is_training):
    """Dilated convolution."""
    return model(images, num_classes, is_training, rate=2)


def model_v0(images, num_classes, is_training):
    """Original model."""
    return model(images, num_classes, is_training, rate=1)


def model(images, num_classes, is_training, rate):
    """Generic model.

  Args:
    images: the input patches, a tensor of size [batch_size, patch_width,
      patch_width, 1].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    rate: Integer, convolution rate. 1 for standard convolution, > 1 for dilated
      convolutions.

  Returns:
    the output logits, a tensor of size [batch_size, 11].

  """
    # Adds a convolutional layer with 32 filters of size [5x5], followed by
    # the default (implicit) Relu activation.
    net = tensorflow.contrib.slim.conv2d(images, 32, [5, 5], padding='SAME', scope='conv1')

    # Adds a [2x2] pooling layer with a stride of 2.
    net = tensorflow.contrib.slim.max_pool2d(net, [2, 2], 2, scope='pool1')

    # Adds a convolutional layer with 64 filters of size [5x5], followed by
    # the default (implicit) Relu activation.
    net = tensorflow.contrib.slim.conv2d(net, 64, [5, 5], padding='SAME', scope='conv2', rate=rate)

    # Adds a [2x2] pooling layer with a stride of 2.
    net = tensorflow.contrib.slim.max_pool2d(net, [2, 2], 2, scope='pool2')

    # Reshapes the hidden units such that instead of 2D maps, they are 1D vectors:
    net = tensorflow.contrib.slim.flatten(net)

    # Adds a fully-connected layer with 1024 hidden units, followed by the default
    # Relu activation.
    net = tensorflow.contrib.slim.fully_connected(net, 1024, scope='fc3')

    # Adds a dropout layer during training.
    net = tensorflow.contrib.slim.dropout(net, 0.5, is_training=is_training, scope='dropout3')

    # Adds a fully connected layer with 'num_classes' outputs. Note
    # that the default Relu activation has been overridden to use no activation.
    net = tensorflow.contrib.slim.fully_connected(net, num_classes, activation_fn=None, scope='fc4')

    return net


def ranked_probability_score(predictions, targets, dim, name=None):
    r"""Calculate the Ranked Probability Score (RPS).

  RPS is given by the formula

    sum_{k=1}^K (CDF_{prediction,k} - CDF_{target,k}) ^ 2

  where CDF denotes the emperical CDF and each value of `k` denotes a different
  class, in rank order. The range of possible RPS values is `[0, K - 1]`, where
  `K` is the total number of classes. Perfect predictions have a score of zero.

  This is a better metric than cross-entropy for probabilistic classification of
  ranked targets, because it penalizes wrong guesses more harshly if they
  predict a target that is further away. For deterministic predictions (zero
  or one) ranked probability score is equal to absolute error in the number of
  classes.

  Importantly (like cross entropy), it is a strictly proper score rule: the
  highest expected reward is obtained by predicting the true probability
  distribution.

  For these reasons, it is widely used for evaluating weather forecasts, which
  are a prototypical use case for probabilistic regression.

  References:
    Murphy AH. A Note on the Ranked Probability Score. J. Appl. Meteorol. 1971,
    10:155-156.
    http://dx.doi.org/10.1175/1520-0450(1971)010<0155:ANOTRP>2.0.CO;2

  Args:
    predictions: tf.Tensor with probabilities for each class.
    targets: tf.Tensor with one-hot encoded targets.
    dim: integer dimension which corresponds to different classes in both
      ``predictions`` and ``targets``.
    name: optional string name for the operation.

  Returns:
    tf.Tensor with the ranked probability score.

  Raises:
    ValueError: if predictions and targets do not have the same shape.
  """
    with tensorflow.name_scope(name, 'ranked_probability_score', [predictions,
                                                                  targets]) as scope:
        predictions = tensorflow.convert_to_tensor(predictions, name='predictions')
        targets = tensorflow.convert_to_tensor(targets, name='targets')

        if not predictions.get_shape().is_compatible_with(targets.get_shape()):
            raise ValueError('predictions and targets must have compatible shapes')

        if predictions.dtype.is_floating and targets.dtype.is_integer:
            # it's safe to coerce integer targets to float dtype
            targets = tensorflow.cast(targets, dtype=predictions.dtype)

        cdf_pred = tensorflow.cumsum(predictions, dim)
        cdf_target = tensorflow.cumsum(targets, dim)

        values = (cdf_pred - cdf_target) ** 2

        # If desired, we could add arbitrary weighting in this sum along dim.
        # That would still be a proper scoring rule (it's equivalent to rescaling
        # the discretization):
        # https://www.stat.washington.edu/research/reports/2008/tr533.pdf
        rps = tensorflow.reduce_sum(values, dim, name=scope)

        return rps