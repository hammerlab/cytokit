import tensorflow as tf
import logging
import os

# See https://github.com/tensorflow/tensorflow/issues/1258
# for interpretations of TF_CPP_MIN_LOG_LEVEL settings
CPP_LEVEL_INFO = '0'
CPP_LEVEL_WARN = '1'
CPP_LEVEL_ERROR = '2'
CPP_LEVEL_FATAL = '3'
CPP_LEVEL_MAP = {
    logging.DEBUG: CPP_LEVEL_INFO, 
    logging.INFO: CPP_LEVEL_INFO,
    logging.WARN: CPP_LEVEL_WARN,
    logging.ERROR: CPP_LEVEL_ERROR,
    logging.FATAL: CPP_LEVEL_FATAL
}


def log_level_code(ll):
    """Resolve string names for log levels to integer codes
    
    Args:
        ll: Log level as string or code; if this is something other than a string
            it will be returned as is and if it is a string it will be converted
            to an integer using whitespace and case insensitive lookups
    Returns:
        Integer code corresponding to logging level
    """
    return logging.getLevelName(ll.strip().upper()) if isinstance(ll, str) else ll


def init_tf_logging(cpp_log_level=logging.WARN, py_log_level=logging.WARN):
    """Set TensorFlow logging levels for Python and C++"""
    py_log_level = log_level_code(py_log_level)
    cpp_log_level = CPP_LEVEL_MAP.get(log_level_code(cpp_log_level), CPP_LEVEL_WARN)
    var = 'TF_CPP_MIN_LOG_LEVEL'
    os.environ[var] = os.getenv(var, cpp_log_level)  # Override only if not set already
    tf.logging.set_verbosity(py_log_level)


def tf_print(t, transform=None):
    """Inject graph operation to print a tensors underlying value (or transformation of it)"""
    def log_value(x):
        print('{} - {}'.format(t.name, x if transform is None else transform(x)))
        return x
    log_op = tf.py_func(log_value, [t], [t.dtype], name=t.name.split(':')[0])[0]
    with tf.control_dependencies([log_op]):
        r = tf.identity(t)
    return r
