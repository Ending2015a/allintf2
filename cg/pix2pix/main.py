# --- built in ---
import os
import re
import sys
import glob
import time
import inspect
import logging
import argparse
import datetime

#from typing import Any
#from typing import List
#from typing import Tuple
#from typing import Union
#from typing import Optional

# --- 3rd party ---
import numpy as np
import tensorflow as tf

#from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops

from matplotlib import pyplot as plt

# --- my module ---
sys.path.append('../../')
import logger


'''
TensorFlow 2.0 implementation of pix2pix network

The implementation follows the original paper version of pix2pix, introduced by Phillip Isola et al.

Reference:
    The original paper: Image-to-Image Translation with Conditional Adversarial Networks
    Arxiv: https://arxiv.org/abs/1611.07004
    Github: https://github.com/phillipi/pix2pix
'''

# === GPU settings ===
os.environ['CUDAVISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# === Logging settings ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DEFAULT_LOGGING_LEVEL = 'DEBUG'


# === Hyper-parameters ===
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNEL_FIRST = False  # NCHW (True) or NHWC (False)

INFERENCE = False
INPUT_PATH = None
OUTPUT_PATH = None

TRAIN = False
BATCH_SIZE = 1  # Training batch size
LAMBDA = 100
GEN_LR = 2e-4
DIS_LR = 2e-4
EPOCHS = 100
EVAL_EPOCHS = 1
SAVE_EPOCHS = 10
EXPORT_BEST = False
EXPORT_LATEST = True

DATA_FILENAME = 'facades.tar.gz'
DATA_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
DATA_ROOT = 'datasets'
TRAIN_PATH = 'facades/train'
TEST_PATH = 'facades/val'
TRAIN_FILE = '*.jpg'
TEST_FILE = '*.jpg'

SEED=None
MODEL_DIR = None
MODEL_NAME = None
LOG_PATH = None
LOG_LEVEL = 'INFO'
LOG = None
VERBOSE = False

BUFFER_SIZE = 100

def make_timestamp(dtime='now', fmt='%Y-%m-%d_%H.%M.%S'):
    '''
    Make timestamps in specified format
    '''

    if dtime == 'now':
        return datetime.datetime.now().strftime(fmt)

    assert isinstance(dtime, datetime), 'dtime must be \'now\' or datetime object'

    return dtime.strftime(fmt)

def parse_args():

    # create timestamp
    day_timestamp = make_timestamp(fmt='%Y-%m-%d')
    sec_timestamp = make_timestamp(fmt='%Y-%m-%d_%H.%M.%S')

    # create parser
    parser = argparse.ArgumentParser(description='Pix2Pix')
    
    # model parameters
    parser.add_argument('--image_height', type=int, help='The height of images', default=IMAGE_HEIGHT)
    parser.add_argument('--image_width', type=int, help='The width of images', default=IMAGE_WIDTH)
    parser.add_argument('--channel_first', help='Whether to use channel first representation', action='store_true')

    # inference parameters
    parser.add_argument('-i', '--input', dest='input_path', type=str, help='Input path', default=None)
    parser.add_argument('-o', '--output', dest='output_path', type=str, help='Output path', default=None)

    # training parameters
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=BATCH_SIZE)
    parser.add_argument('--lambda', dest='gen_lambda', type=float, help='The factor for generator L1-loss', default=LAMBDA)
    parser.add_argument('--gen_lr', type=float, help='The learning rate of Adam for the generator', default=GEN_LR)
    parser.add_argument('--dis_lr', type=float, help='The learning rate of Adam for the discriminator', default=DIS_LR)
    parser.add_argument('--epochs', type=int, help='Training epochs', default=EPOCHS)
    parser.add_argument('--eval_epochs', type=int, help='Evaluate every N epochs', default=EVAL_EPOCHS)
    parser.add_argument('--save_epochs', type=int, help='Save model for every N epochs', default=SAVE_EPOCHS)
    parser.add_argument('--export_best', help='Whether to export the best model', action='store_true')

    # dataset parameters
    parser.add_argument('--data_filename', type=str, help='The filename of the dataset', default=DATA_FILENAME)
    parser.add_argument('--data_url', type=str, help='The url of the dataset', default=DATA_URL)
    parser.add_argument('--data_root', type=str, help='The root path to store the downloaded datasets', default=DATA_ROOT)
    parser.add_argument('--train_path', type=str, help='The path to the training set', default=TRAIN_PATH)
    parser.add_argument('--train_file', type=str, help='The filename of the training set', default=TRAIN_FILE)
    parser.add_argument('--test_path', type=str, help='The path to the testing set', default=TEST_PATH)
    parser.add_argument('--test_file', type=str, help='The filename of the testing set', default=TEST_FILE)

    # other settings
    parser.add_argument('--seed', type=int, help='Random seed', default=None)
    parser.add_argument('--model_dir', type=str, help='The model directory, default: ./model/ckpt-{timestamp}', 
                                                  default='./model/ckpt-{}'.format(day_timestamp))
    parser.add_argument('--model_name', type=str, help='The name of the model', default='model')
    parser.add_argument('--log_path', type=str, help='The logging path, default: {model_dir}/pix2pix-{timestamp}.log', default=LOG_PATH)
    parser.add_argument('--log_level', type=str, help='The logging level, must be one of [\'DEBUG\', \'INFO\', \'WARNING\']', 
                                                  default=LOG_LEVEL)
    parser.add_argument('--verbose', help='If True, more loggin is printed', action='store_true')

    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = os.path.join(args.model_dir, 'log/pix2pix-' + sec_timestamp) + '.log'

    return args

def apply_hyperparameters(args):
    global IMAGE_HEIGHT
    global IMAGE_WIDTH
    global CHANNEL_FIRST

    global INFERENCE
    global INPUT_PATH
    global OUTPUT_PATH

    global TRAIN
    global BATCH_SIZE
    global LAMBDA
    global GEN_LR
    global DIS_LR
    global EPOCHS
    global EVAL_EPOCHS
    global SAVE_EPOCHS
    global EXPORT_BEST
    global EXPORT_LATEST

    global DATA_FILENAME
    global DATA_URL
    global DATA_ROOT 
    global TRAIN_PATH 
    global TEST_PATH
    global TRAIN_FILE
    global TEST_FILE

    global SEED
    global MODEL_DIR
    global MODEL_NAME
    global LOG_PATH
    global LOG_LEVEL
    global LOG
    global VERBOSE
    

    # assign hyperparameters
    IMAGE_HEIGHT = args.image_height
    IMAGE_WIDTH = args.image_width
    CHANNEL_FIRST = args.channel_first

    INFERENCE = True if args.input_path else False
    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path

    TRAIN = args.train
    BATCH_SIZE = args.batch_size
    LAMBDA = args.gen_lambda
    GEN_LR = args.gen_lr
    DIS_LR = args.dis_lr
    EPOCHS = args.epochs
    EVAL_EPOCHS = args.eval_epochs
    SAVE_EPOCHS = args.save_epochs
    EXPORT_BEST = args.export_best
    EXPORT_LATEST = not args.export_best

    DATA_FILENAME = args.data_filename
    DATA_URL = args.data_url
    DATA_ROOT = args.data_root
    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path
    TRAIN_FILE = args.train_file
    TEST_FILE = args.test_file

    SEED = args.seed
    MODEL_DIR = args.model_dir
    MODEL_NAME = args.model_name
    LOG_PATH = args.log_path
    LOG_LEVEL = args.log_level
    VERBOSE = args.verbose

    if INFERENCE and TRAIN:
        raise ValueError('Input file specified for training mode is not allowed')

    # output path not specified
    if INFERENCE and OUTPUT_PATH is None:
        fname = os.path.basename(INPUT_PATH)
        name, ext = os.path.splitext(fname)
        OUTPUT_PATH = './{}_generated' + ext

    # fixed random seed if specified
    if args.seed is not None:
        tf.random.set_seed(SEED)
        np.random.seed(SEED)

    # create logging path
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    # apply loggin settings
    logger.Config.Use(filename=LOG_PATH, level=LOG_LEVEL, colored=True, reset=False)
    # create logger
    LOG = logger.getLogger('main')

    # ====== print args
    LOG.set_header('Arguments')

    LOG.subgroup('model')
    LOG.add_row('Image height', IMAGE_HEIGHT)
    LOG.add_row('Image width', IMAGE_WIDTH)
    LOG.add_row('Channel first', CHANNEL_FIRST)

    if TRAIN:
        # training mode
        LOG.subgroup('training')
        LOG.add_row('Training', TRAIN)
        LOG.add_row('Batch size', BATCH_SIZE)
        LOG.add_row('Lambda', LAMBDA)
        LOG.add_row('Generator lr', GEN_LR)
        LOG.add_row('Discriminator lr', DIS_LR)
        LOG.add_row('Eval epochs', EVAL_EPOCHS)
        LOG.add_row('Save epochs', SAVE_EPOCHS)
        LOG.add_row('Export best', EXPORT_BEST)
        LOG.add_row('Export latest', EXPORT_LATEST)
    
    if not INFERENCE:
        # training or testing mode
        LOG.subgroup('dataset')
        LOG.add_row('Data filename', DATA_FILENAME)
        LOG.add_row('Data URL', DATA_URL)
        LOG.add_row('Data root', DATA_ROOT)
        LOG.add_row('Training set path', os.path.join(TRAIN_PATH, TRAIN_FILE))
        LOG.add_row('Testing set path', os.path.join(TEST_PATH, TEST_FILE))

    if INFERENCE:
        # inference mode
        LOG.subgroup('inference')
        LOG.add_row('Input path', INPUT_PATH)
        LOG.add_row('Output path', OUTPUT_PATH)

    LOG.subgroup('others')
    LOG.add_row('Random seed', SEED)
    LOG.add_row('Model directory', MODEL_DIR)
    LOG.add_row('Model name', MODEL_NAME)
    LOG.add_row('Logging path', LOG_PATH)
    LOG.add_row('Logging level', LOG_LEVEL)
    LOG.add_row('Verbose', VERBOSE)

    LOG.flush('INFO')
    # ====================

if __name__ == '__main__':
    # set hyperparameters
    apply_hyperparameters(parse_args())

def verify_settings_2d(s):
    '''
    Used in verifying any 2d layer configurations, 
    e.g. kernel size, strides, paddings
    '''
    
    if s is None:
        return None, None
    
    if isinstance(s, int):
        return s, s
    else:
        assert isinstance(s, (tuple, list)) and len(s) == 2

        return s[0], s[1]


def infer_deconv_output_length(input_length,
                               size,
                               padding,
                               output_padding=None,
                               stride=0,
                               dilation=1):
    '''
    Compute the output size of the deconvolution layer 
    with parameter settings specified
    '''

    assert padding in {'SAME', 'VALID', 'FULL'}
    if input_length is None:
        return None

    # dilated kernel size
    size = size + (size - 1) * (dilation - 1)

    if output_padding is None:
        if padding == 'VALID':
            length = input_length * stride + max(size - stride, 0)
        elif padding == 'FULL':
            length = input_length * stride - (stride + size - 2)
        elif padding == 'SAME':
            length = input_length * stride

    else:
        if padding == 'SAME':
            pad = size // 2
        elif padding == 'VALID':
            pad = 0
        elif padding == 'FULL':
            pad = size - 1

        length = ((input_length - 1) * stride + size - 2 * pad + output_padding)

    return length


def get_initializer_by_type(*args, type=None, **kwargs):
    '''
    Create initializer

    Args:
        type: (Optional[tf.initializers.Initializer]) the type of initializer
    '''

    if len(args) > 0:

        assert len(args) == 1

        if isinstance(args[0], str):
            assert args[0] in ['zeros', 'ones']

            if args[0] == 'zeros':
                return tf.initializers.Zeros()
            elif args[0] == 'ones':
                return tf.initializers.Ones()
        
        elif isinstance(args[0], tf.initializers.Initializer):
            return args[0]

        else:
            if type is not None:
                return type(*args, **kwargs)

            return tf.initializers.Constant(args)
    
    if type is not None:
        return type(**kwargs)
            
    raise ValueError('Initializer type not specified')


def get_initializer(*args, **kwargs):
    '''
    Create initializers

    Args:
        args[0]: (int, float, str, tf.initializers.Initializer)
            int, float: Constnat initializer
            str: ('zeros', 'ones')
            tf.initializers.Initializer: initializer
    
    Kwargs:
        gain: (float)

    '''

    # default initializer
    if len(args) == 0 and len(kwargs) == 0:
        # no params
        return tf.initializer.GlorotNormal()

    assert (len(args) > 0 and len(kwargs) == 0) or (len(kwargs) > 0 and len(args) == 0)


    if len(args) > 0:
        
        return get_initializer_by_type(args[0])

    if 'gain' in kwargs:
        if isinstance(kwargs['gain'], tf.initializers.Initializer):
            return kwargs['gain']
        else:
            return get_initializer_by_type(gain=kwargs['gain'], type=tf.initializers.Orthogonal)
    
    if 'bias' in kwargs:
        if isinstance(kwargs['bias'], tf.initializers.Initializer):
            return kwargs['bias']
        else:
            return get_initializer_by_type(kwargs['bias'])

    if 'seed' in kwargs:
        return tf.initializer.GlorotNormal(seed=seed)

    raise ValueError('Unknown initializer options: {}'.format(kwargs))


def auto_naming(self, name=None):
    '''
    Create unique name for each module automatically
    '''
    
    # create name dictionary
    if not hasattr(auto_naming, 'name_dict'):
        setattr(auto_naming, 'name_dict', {})

    def to_snake_case(name):
        intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
        insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()

        if insecure[0] != '_':
            return insecure
        return 'private' + insecure
    
    # get default name
    if name is None:
        name = to_snake_case(self.__class__.__name__)

    # create key
    key = name

    # get current name scope. 
    # I don't know why tf.compat.v1.get_default_graph().get_name_scope() doesn't work,
    # so I use tensorflow.python.framework.ops.get_name_scope(), instead. This function
    # is not officially documented.
    name_scope = ops.get_name_scope()
    if name_scope:
        key = os.path.join(name_scope, key)

    # name does not exist
    if key not in auto_naming.name_dict:
        auto_naming.name_dict[key] = 0
    
    # name already exists
    else:
        # increate counter
        auto_naming.name_dict[key] += 1
        name = '{}_{}'.format(name, auto_naming.name_dict[key])

    return name

# === Primitive Modules ===


class Conv(tf.Module):
    def __init__(self, n_kernel,
                       size,
                       stride,
                       gain=tf.initializers.RandomNormal(0.0, 0.02),
                       bias=0.0,
                       dilations=1,
                       padding='SAME',
                       is_biased=False,
                       channel_first=CHANNEL_FIRST,
                       name=None):
        '''
        2D Convolution

        Args:
            n_kernel: (int) the number of kernels
            size: (int, [int, int]) size of the kernel (height, width)
            stride: (int, [int, int]) convolution stride (height, width)
            gain: (float) for Orthogonal initializer
            bias: (float) bias
            dilations: (Tuple[int, int]) dilation rate
            padding: (str) padding
            is_biased: (bool) whether to add bias after the convolution
            channel_first: (bool) data format
            name: (Optional[str]) module name
        '''

        super(Conv, self).__init__(name=auto_naming(self, name))

        # variables
        self.w = None
        self.b = None

        self.n_kernel = n_kernel
        self.size = verify_settings_2d(size)
        self.stride = verify_settings_2d(stride)
        self.gain = gain
        self.bias = bias
        self.dilations = verify_settings_2d(dilations)
        self.padding = padding
        self.is_biased = True if is_biased else False
        self.channel_first = True if channel_first else False

        self.w_initializer = get_initializer(gain=gain)
        self.b_initializer = get_initializer(bias)

        self.has_built = False
        self.data_format = None
        self._stride = None


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # build module
        if not self.has_built:

            # get kernel size, stride(height, width)
            k_h, k_w = self.size
            s_h, s_w = self.stride

            # check whether NCHW or NHWC
            if self.channel_first:
                self.data_format = 'NCHW'
                channel_axis = 1
                stride = [1, 1, s_h, s_w]
            else:
                self.data_format = 'NHWC'
                channel_axis = 3
                stride = [1, s_h, s_w, 1]

            self._stride = stride

            # input channels
            c_i = input.shape[channel_axis]
            # output channels
            c_o = self.n_kernel

            # create weight shape
            w_shape = [k_h, k_w, c_i, c_o]

            # create weights
            self.w = tf.Variable(self.w_initializer(w_shape, dtype=tf.float32),
                                 trainable=training,
                                 dtype=tf.float32,
                                 name='w')


            if self.is_biased:
                # create bias shape
                b_shape = [c_o]

                # create bias
                self.b = tf.Variable(self.b_initializer(b_shape, dtype=tf.float32),
                                    trainable=training,
                                    dtype=tf.float32,
                                    name='b')

            
            
        
        # perform convolution
        output = tf.nn.conv2d(input, self.w,
                                     strides=self._stride,
                                     padding=self.padding,
                                     data_format=self.data_format,
                                     dilations=self.dilations)

        
        if self.is_biased:
            
            # apply bias
            output = tf.nn.bias_add(output, self.b, data_format=self.data_format)

        
        # print logging
        if not self.has_built:
            
            LOG.set_header('Conv2D \'{}\''.format(self.name))
            
            LOG.add_row('scope', '{}'.format(self.name_scope.name))
            LOG.add_row('training', training)

            LOG.subgroup('shape')
            LOG.add_row('input shape', input.shape)
            LOG.add_row('output shape', output.shape)

            if VERBOSE:
                LOG.subgroup('config')
                LOG.add_row('n_kernel', self.n_kernel)
                LOG.add_row('size', self.size)
                LOG.add_row('stride', self.stride)
                LOG.add_row('gain', self.gain)
                LOG.add_row('bias', self.bias)
                LOG.add_row('dilations', self.dilations)
                LOG.add_row('padding', self.padding)
                LOG.add_row('is biased', self.is_biased)
            

            LOG.flush('DEBUG')

            self.has_built = True

        return output



class Deconv(tf.Module):
    def __init__(self, n_kernel,
                       size,
                       stride,
                       gain=tf.initializers.RandomNormal(0.0, 0.02),
                       bias=0.0,
                       dilations=1,
                       padding='SAME',
                       output_padding=None,
                       is_biased=False, 
                       channel_first=CHANNEL_FIRST,
                       name=None):
        '''
        2D Convolution

        Args:
            n_kernel: (int) the number of kernels
            size: (int, [int, int]) size of the kernel (height, width)
            stride: (int, [int, int]) convolution stride (height, width)
            gain: (float) for Orthogonal initializer
            bias: (float) bias
            output_padding: (Optional[int]) padding size
            is_biased: (bool) whether to add bias after the convolution
            channel_first: (bool) data format
            name: (Optional[str]) module name
        '''

        super(Deconv, self).__init__(name=auto_naming(self, name))

        # variables
        self.w = None
        self.b = None

        self.n_kernel = n_kernel
        self.size = verify_settings_2d(size)
        self.stride = verify_settings_2d(stride)
        self.gain = gain
        self.bias = bias
        self.dilations = verify_settings_2d(dilations)
        self.padding = padding
        self.output_padding = verify_settings_2d(output_padding)
        self.is_biased = True if is_biased else False
        self.channel_first = True if channel_first else False

        self.w_initializer = get_initializer(gain=gain)
        self.b_initializer = get_initializer(bias)

        self.has_built = False
        self.data_format = None
        self.output_shape = None
        self._stride = None


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):
        
        # build module
        if not self.has_built:

            # get kernel size, stride, dilations, output padding (height, width)
            k_h, k_w = self.size
            s_h, s_w = self.stride
            d_h, d_w = self.dilations
            p_h, p_w = self.output_padding

            # check whether NCHW or NHWC
            if self.channel_first:
                self.data_format = 'NCHW'
                channel_axis, height_axis, width_axis = 1, 2, 3
                stride = [1, 1, s_h, s_w]
            else:
                self.data_format = 'NHWC'
                height_axis, width_axis, channel_axis = 1, 2, 3
                stride = [1, s_h, s_w, 1]

            self._stride = stride

            # input channels
            c_i = input.shape[channel_axis]
            # output channels
            c_o = self.n_kernel

            # create weight shape
            w_shape = [k_h, k_w, c_o, c_i]

            # create weights
            self.w = tf.Variable(self.w_initializer(w_shape, dtype=tf.float32),
                                 trainable=training,
                                 dtype=tf.float32,
                                 name='w')

            if self.is_biased:

                # create bias shape
                b_shape = [c_o]

                # create bias
                self.b = tf.Variable(self.b_initializer(b_shape, dtype=tf.float32),
                                    trainable=training,
                                    dtype=tf.float32,
                                    name='b')
            
            # infer output shape
            o_h = infer_deconv_output_length(input.shape[height_axis],
                                             k_h,
                                             padding=self.padding,
                                             output_padding=p_h,
                                             stride=s_h,
                                             dilation=d_h)

            o_w = infer_deconv_output_length(input.shape[width_axis],
                                             k_w,
                                             padding=self.padding,
                                             output_padding=p_w,
                                             stride=s_w,
                                             dilation=d_w)

            if self.channel_first:
                self.output_shape = [input.shape[0], c_o, o_h, o_w]
            else:
                self.output_shape = [input.shape[0], o_h, o_w, c_o]



        # perform deconvolution
        output = tf.nn.conv2d_transpose(input, self.w,
                                               output_shape=tf.convert_to_tensor(self.output_shape),
                                               strides=self._stride,
                                               padding=self.padding,
                                               data_format=self.data_format,
                                               dilations=self.dilations)

        output.set_shape(self.output_shape)


        if self.is_biased:

            # apply bias
            output = tf.nn.bias_add(output, self.b, data_format=self.data_format)


        # print logging
        if not self.has_built:
            
            LOG.set_header('Deconv2D \'{}\''.format(self.name))
            
            LOG.add_row('scope', '{}'.format(self.name_scope.name))
            LOG.add_row('training', training)

            LOG.subgroup('shape')
            LOG.add_row('input shape', input.shape)
            LOG.add_row('output shape', output.shape)

            if VERBOSE:
                LOG.subgroup('config')
                LOG.add_row('n_kernel', self.n_kernel)
                LOG.add_row('size', self.size)
                LOG.add_row('stride', self.stride)
                LOG.add_row('gain', self.gain)
                LOG.add_row('bias', self.bias)
                LOG.add_row('dilations', self.dilations)
                LOG.add_row('padding', self.padding)
                LOG.add_row('output_padding', self.output_padding)
                LOG.add_row('is biased', self.is_biased)

            LOG.flush('DEBUG')

            self.has_built = True
        
        return output


class BatchNorm(tf.Module):
    def __init__(self, axis=None,
                       momentum=0.99,
                       epsilon=1e-3,
                       gain='ones',
                       bias='zeros',
                       mean='zeros',
                       var=tf.initializers.Constant(value=0.02),
                       channel_first=CHANNEL_FIRST,
                       name=None):
        '''
        Perform batch normalization

        Args:
            axis: (None, int, list) if None, the axes is generated automatically according to arg `channel_first`
            momentum: (float)
            epsilon: (float)
            gain: (float, str, tf.initializers.Initializer) gamma
            bias: (float, str, tf.initializers.Initializer) beta
            mean: (float, str, tf.initializers.Initializer) initial moving mean
            var: (float, str, tf.initializers.Initializer) initial moving variance
            channel_first: (bool) NHWC or NCHW
            name: (Optional[str]) module name
        '''
        
        super(BatchNorm, self).__init__(name=auto_naming(self, name))

        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.gain = gain
        self.bias = bias
        self.mean = mean
        self.var = var
        self.channel_first = True if channel_first else False

        self.gamma_initializer = get_initializer(gain)
        self.beta_initializer = get_initializer(bias)
        self.mean_initializer = get_initializer(mean)
        self.var_initializer = get_initializer(var)

        
        self.has_built = False
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_var = None
        self.broadcast_shape = None
        self.reduction_axes = None

        # set default axis if axis is None
        if self.axis is None:
            self.axis = 1 if self.channel_first else -1

        # convert to list
        if not isinstance(self.axis, (list, tuple)):
            self.axis = [self.axis]



    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # initialize module
        if not self.has_built:
        
            # get input dimensions
            #ndims = len(input.shape[:])
            ndims = len(input.shape)
            axis = self.axis

            # normalize axis (-1 -> ndims-1)
            axis = sorted(list(set([ndims+x if x<0 else x for x in axis])))
            self.axis = axis

            # create shape
            shape = [input.shape[x] for x in self.axis]

            # create gamma
            self.gamma = tf.Variable(self.gamma_initializer(shape, dtype=tf.float32),
                                     trainable=training,
                                     dtype=tf.float32,
                                     name='gamma')

            # create beta
            self.beta = tf.Variable(self.beta_initializer(shape, dtype=tf.float32),
                                    trainable=training,
                                    dtype=tf.float32,
                                    name='beta')

            # create moving mean
            self.moving_mean = tf.Variable(self.mean_initializer(shape, dtype=tf.float32),
                                           trainable=False,
                                           dtype=tf.float32,
                                           name='mean')

            # create moving var
            self.moving_var = tf.Variable(self.var_initializer(shape, dtype=tf.float32),
                                          trainable=False,
                                          dtype=tf.float32,
                                          name='var')

            # broadcast gamma/beta into the correct shape
            #   for example, if input shape is [N, 256, 256, 3] in data format `NHWC`,
            #   and perform batch norm on axis `C`, then we have reduction_axes = `NHW`,
            #   broadcast_shape = [1, 1, 1, `C`]
            self.reduction_axes = [x for x in range(ndims) if x not in self.axis]
            self.broadcast_shape = [1 if x in self.reduction_axes else input.shape[x] for x in range(ndims)]


        def apply_moving_average(variable, value, momentum):
            '''
                a = a * momentum + v * (1-momentum)
            ->  a = a * (1-decay) + v * decay   #decay = 1-momentum
            ->  a = a - a * decay + v * decay
            ->  a = a - (a - v) * decay
            '''
            decay = tf.convert_to_tensor(1.0 - momentum, dtype=tf.float32, name='decay')

            delta = (variable - value) * decay

            variable.assign_sub(delta=delta, name='apply_moving_average')

        # correct shapes
        scale = tf.reshape(self.gamma, [-1])
        offset = tf.reshape(self.beta, [-1])
        mean = tf.reshape(self.moving_mean, [-1])
        var = tf.reshape(self.moving_var, [-1])

        # compute mean/variance
        #mean, var = tf.nn.moments(tf.convert_to_tensor(input), self.reduction_axes, keepdims=False)


        if training:

            output, mean, var = tf.compat.v1.nn.fused_batch_norm( # is training
                                    tf.convert_to_tensor(input),
                                    scale=scale,
                                    offset=offset,
                                    epsilon=self.epsilon,
                                    data_format='NCHW' if self.channel_first else 'NHWC')
            

            apply_moving_average(self.moving_mean, mean, self.momentum)
            apply_moving_average(self.moving_var, var, self.momentum)

        else:

            output, mean, var = tf.compat.v1.nn.fused_batch_norm( # is not training
                                    tf.convert_to_tensor(input),
                                    scale=scale,
                                    offset=offset,
                                    mean=mean,
                                    variance=var,
                                    epsilon=self.epsilon,
                                    data_format='NCHW' if self.channel_first else 'NHWC',
                                    is_training=False)
            


        if not self.has_built:

            LOG.set_header('BatchNorm \'{}\''.format(self.name))
            
            LOG.add_row('scope', '{}'.format(self.name_scope.name))
            LOG.add_row('training', training)

            LOG.subgroup('shape')
            LOG.add_row('input shape', input.shape)
            LOG.add_row('output shape', output.shape)

            if VERBOSE:
                LOG.subgroup('config')
                LOG.add_row('axis', axis)
                LOG.add_row('momentum', self.momentum)
                LOG.add_row('gain', self.gain)
                LOG.add_row('bias', self.bias)
                LOG.add_row('mean', self.mean)
                LOG.add_row('var', self.var)

            LOG.flush('DEBUG')

            self.has_built = True

        return output


class Dropout(tf.Module):
    def __init__(self, rate,
                       noise_shape=None,
                       seed=None,
                       name=None):

        super(Dropout, self).__init__(name=auto_naming(self, name))

        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

        self.has_built = False


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):



        # if training, use user specified dropout rate, otherwise ignore dropout (rate=0.0)
        rate = self.rate if training else 0.0

        output = tf.nn.dropout(input, rate=rate,
                                      noise_shape=self.noise_shape,
                                      seed=self.seed)

        # print logging
        if not self.has_built:
            LOG.set_header('Dropout \'{}\''.format(self.name))
            
            LOG.add_row('scope', '{}'.format(self.name_scope.name))
            LOG.add_row('training', training)

            LOG.subgroup('shape')
            LOG.add_row('input shape', input.shape)
            LOG.add_row('output shape', output.shape)

            if VERBOSE:
                LOG.subgroup('config')
                LOG.add_row('rate', self.rate)
                LOG.add_row('noise_shape', self.noise_shape)
                LOG.add_row('seed', self.seed)

            LOG.flush('DEBUG')

            self.has_built = True

        
        return output
    

class Padding(tf.Module):
    def __init__(self, padding=1,
                       mode='CONSTANT', 
                       constant_values=0,
                       channel_first=CHANNEL_FIRST,
                       name=None):
        
        super(Padding, self).__init__(name=auto_naming(self, name))

        self.padding = verify_settings_2d(padding)
        self.mode = mode
        self.constant_values = constant_values
        self.channel_first = channel_first

        self.has_built = False
        self.pad_shape = None


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # build module
        if not self.has_built:

            #ndims = len(input.shape[:])
            ndims = len(input.shape)

            assert ndims == 4

            p_h, p_w = self.padding

            # find height/width axis
            if self.channel_first: # NCHW
                self.pad_shape = [ [0, 0], [0, 0], [p_h, p_h], [p_w, p_w] ]
            else: # NHWC
                self.pad_shape = [ [0, 0], [p_h, p_h], [p_w, p_w], [0, 0] ]


        output = tf.pad(input, paddings=self.pad_shape,
                               mode=self.mode, 
                               constant_values=self.constant_values)

        # print logging
        if not self.has_built:
            LOG.set_header('Padding \'{}\''.format(self.name))
            
            LOG.add_row('scope', '{}'.format(self.name_scope.name))
            LOG.add_row('training', training)

            LOG.subgroup('shape')
            LOG.add_row('input shape', input.shape)
            LOG.add_row('output shape', output.shape)

            if VERBOSE:
                LOG.subgroup('config')
                LOG.add_row('padding', self.padding)
                LOG.add_row('mode', self.mode)
                LOG.add_row('constant_values', self.constant_values)
                LOG.add_row('pad_shape', self.pad_shape)

            LOG.flush('DEBUG')

            self.has_built = True
        
        return output

# === Sub Modules ===

class DownSample(tf.Module):
    def __init__(self, n_kernel,
                       size,
                       stride=2,
                       apply_batchnorm=True,
                       apply_activ=True,
                       name=None):

        super(DownSample, self).__init__(name=auto_naming(self, name))

        self.apply_batchnorm = apply_batchnorm
        self.apply_activ = apply_activ

        # create convolution layer
        self.conv = Conv(n_kernel=n_kernel,
                         size=size,
                         stride=stride,
                         padding='SAME',
                         is_biased=False)

        # create batch normalization layer
        if self.apply_batchnorm:
            self.batchnorm = BatchNorm()


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):
        # forward convolution
        output = self.conv(input, training=training)

        # forward batch normalization
        if self.apply_batchnorm:
            output = self.batchnorm(output, training=training)
        
        # apply leaky relu
        if self.apply_activ:
            output = tf.nn.leaky_relu(output, alpha=0.2)

        return output


class UpSample(tf.Module):
    def __init__(self, n_kernel, 
                       size,
                       stride=2,
                       apply_batchnorm=True,
                       apply_dropout=False,
                       apply_activ=True,
                       name=None):

        super(UpSample, self).__init__(name=auto_naming(self, name))

        self.apply_batchnorm = apply_batchnorm
        self.apply_activ = apply_activ
        self.apply_dropout = apply_dropout

        # create deconvolution layer
        self.deconv = Deconv(n_kernel=n_kernel,
                           size=size,
                           stride=stride,
                           padding='SAME',
                           is_biased=False)

        # create batch normalization layer
        if self.apply_batchnorm:
            self.batchnorm = BatchNorm()

        # create dropout layer
        if self.apply_dropout:
            self.dropout = Dropout(0.5)


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # forward deconvolution
        output = self.deconv(input, training=training)
        
        # forward batch normalization
        if self.apply_batchnorm:
            output = self.batchnorm(output, training=training)

        # apply dropout
        if self.apply_dropout:
            output = self.dropout(output, training=training)

        # apply relu
        if self.apply_activ:
            output = tf.nn.relu(output)

        return output



class UNet(tf.Module):
    def __init__(self, name=None):
        super(UNet, self).__init__(name=auto_naming(self, name))

        self.downsample_spec = [dict(n_kernel=64,  size=4, apply_batchnorm=False),
                                dict(n_kernel=128, size=4),
                                dict(n_kernel=256, size=4),
                                dict(n_kernel=512, size=4),
                                dict(n_kernel=512, size=4),
                                dict(n_kernel=512, size=4),
                                dict(n_kernel=512, size=4),
                                dict(n_kernel=512, size=4, apply_batchnorm=False, apply_activ=False) ]

        self.upsample_spec = [dict(n_kernel=512, size=4, apply_dropout=True),
                              dict(n_kernel=512, size=4, apply_dropout=True),
                              dict(n_kernel=512, size=4, apply_dropout=True),
                              dict(n_kernel=512, size=4),
                              dict(n_kernel=256, size=4),
                              dict(n_kernel=128, size=4),
                              dict(n_kernel=64, size=4),
                              dict(n_kernel=3, size=4, apply_batchnorm=False, apply_activ=False)]

        # create sub-modules under the name scope
        with self.name_scope:
            self.downsamples = [ DownSample(**spec) for spec in self.downsample_spec ]
            self.upsamples = [ UpSample(**spec) for spec in self.upsample_spec ]


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        outputs = []

        # UNet down sampling
        for layer in self.downsamples:
            # forward downsample layers  
            output = layer(input, training=training)
            # append forwarded results to the TensorArray
            outputs.append(output)
            # prepare inputs for the next layer
            input = output


        outputs.reverse()


        # UNet up sampling
        for idx in range(len(self.upsamples)):
            # prepare inputs, concatenate input from the previous layer with the 
            # output feature maps from the corresponding down sampling layers, except
            # for the first up sampling layer, since its corresponding layer is the
            # previous layer.
            if idx > 0:
                input = tf.concat([input, outputs[idx]], axis=-1)
            else:
                input = tf.nn.relu(input)

            output = self.upsamples[idx](input, training=training)

            input = output

        # final activation function, normalize output between [-1, 1]
        output = tf.nn.tanh(input)

        return output


# === Main Modules ===

class Generator(tf.Module):
    def __init__(self, name=None):
        super(Generator, self).__init__(name=auto_naming(self, name))

        # create sub-modules under the name scope
        with self.name_scope:
            self.net = UNet()

    @tf.function
    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        output = self.net(input, training=training)

        return output

    @tf.function
    def loss(self, y, y_gen, x_pred):
        '''
        generator minimize L(G) = -E[log(D(G(x)))] + lambda * L1(y-G(x))

        Args:
            y: y
            y_gen: G(x)
            x_pred: D(G(x))

        Reference:
            https://developers.google.com/machine-learning/gan/loss#modified-minimax-loss
        '''

        gan_loss = tf.math.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.ones_like(x_pred), logits=x_pred))

        l1_loss = tf.math.reduce_mean(tf.math.abs(y - y_gen))

        loss = gan_loss + LAMBDA * l1_loss

        return loss


class Discriminator(tf.Module):
    def __init__(self, name=None):
        super(Discriminator, self).__init__(name=auto_naming(self, name))

        # create sub-modules under the name scope
        with self.name_scope:
            self.down1 = DownSample(64, 4, apply_batchnorm=False)
            self.down2 = DownSample(128, 4)
            self.down3 = DownSample(256, 4)
            
            self.pad1 = Padding()
            self.down4 = DownSample(512, 4, stride=1)

            self.pad2 = Padding()
            self.conv2 = Conv(1, 4, stride=1, is_biased=True)

    @tf.function
    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # concatenate multiple inputs
        input = tf.concat(input, axis=-1)

        output = self.down1(input, training=training)
        output = self.down2(output, training=training)
        output = self.down3(output, training=training)

        output = self.pad1(output, training=training)
        output = self.down4(output, training=training)

        output = self.pad2(output, training=training)
        output = self.conv2(output, training=training)

        return output

    @tf.function
    def loss(self, y_pred, x_pred):
        '''
        discriminator maximize L(G, D) = E[log(D(y))] + E[log(1-D(G(x)))]

        Args:
            y_pred: D(y)
            x_pred: D(G(x))

        Reference:
            https://developers.google.com/machine-learning/gan/loss#minimax-loss
        '''

        real_loss = tf.math.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.ones_like(y_pred), logits=y_pred))

        gened_loss = tf.math.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.zeros_like(x_pred), logits=x_pred))

        loss = real_loss + gened_loss
        return loss


def maybe_download_dataset(fname,
                           origin,
                           extract=False,
                           file_hash=None,
                           hash_algorithm='auto',
                           cache_subdir=os.path.abspath(DATA_ROOT),
                           archive_format='auto'):

    return tf.keras.utils.get_file(fname=fname,
                                   origin=origin,
                                   extract=extract,
                                   file_hash=file_hash,
                                   hash_algorithm=hash_algorithm,
                                   cache_subdir=cache_subdir,
                                   archive_format=archive_format)
    

def create_dataset(path, is_train=True):
    
    def process_path(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32) * 2.0 - 1.0
        return img

    def process_image(is_train):
        if is_train:
            def _process_data(img):
                # slice image
                real, fake = tf.split(img, [IMAGE_WIDTH, IMAGE_WIDTH], axis=1, num=2)
                #real = tf.slice(img, [0, 0, 0], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                #fake = tf.slice(img, [0, IMAGE_WIDTH, 0], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                # resize
                real = tf.image.resize(real, [int(IMAGE_HEIGHT*1.12), int(IMAGE_WIDTH*1.12)])
                fake = tf.image.resize(fake, [int(IMAGE_HEIGHT*1.12), int(IMAGE_WIDTH*1.12)])
                
                # stack [2, H, W, C]
                stacked = tf.stack([real, fake], axis=0)
                # random crop
                cropped = tf.image.random_crop(stacked, size=[2, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                
                # unpack images
                real, fake = tf.unstack(cropped, num=2, axis=0)

                # random flip
                if np.random.random() > 0.5:
                    real = tf.image.flip_left_right(real)
                    fake = tf.image.flip_left_right(fake)

                return tf.concat([fake, real], axis=2)
        else:
            def _process_data(img):
                # slice image
                real, fake = tf.split(img, [IMAGE_WIDTH, IMAGE_WIDTH], axis=1, num=2)
                #real = tf.slice(img, [0, 0, 0], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                #fake = tf.slice(img, [0, IMAGE_WIDTH, 0], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                # resize
                real = tf.image.resize(real, [IMAGE_HEIGHT, IMAGE_WIDTH])
                fake = tf.image.resize(fake, [IMAGE_HEIGHT, IMAGE_WIDTH])

                return tf.concat([fake, real], axis=2)

        return _process_data

    # list files
    # since tf.data.Dataset.list_files can not sort the paths, use glob.glob instead.
    #list_ds = tf.data.Dataset.list_files(path)
    list_ds = tf.data.Dataset.from_tensor_slices(sorted(glob.glob(path)))
    # process file paths to image
    img_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # process images 
    ds = img_ds.map(process_image(is_train), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if is_train:
        ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    else:
        # test set, do not shuffle
        ds = ds.batch(BATCH_SIZE)

    return ds

# === Inference ===

def inference(input, gen, preprocess=True):
    '''
    Args:
        input: input image (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    '''

    # get shape
    ndim = len(input.shape)
    if ndim == 3:
        input_shape = input.shape[0:2] # H, W
    elif ndim == 4:
        input_shape = input.shape[1:3] # H, W
    else:
        raise ValueError('Unknown shape of image with dimention: {}, only accept 3D or 4D images'.format(ndim))

    input_dtype = input.dtype

    if preprocess:
        # convert image type [-1.0, 1.0]
        
        input = tf.image.convert_image_dtype(input, tf.float32) * 2.0 - 1.0

    # resize image
    x = tf.image.resize(input, [IMAGE_HEIGHT, IMAGE_WIDTH])
    x_shape = x.shape

    # reshape to 4D image
    x = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    # generate
    y_gen = gen(x, training=False)
    # reshape to the original shape
    y_gen = tf.reshape(y_gen, x_shape)

    # resize to the original size
    y_gen = tf.image.resize(y_gen, input_shape)

    if preprocess:
        # convert to original dtype
        y_gen = tf.image.convert_image_dtype(y_gen*0.5+0.5, input_dtype)

    return y_gen.numpy()

# === Testing phase ===

@tf.function
def test_step(input, gen, dis):
    # x=input, y=target (real)
    x, y = tf.split(input, num_or_size_splits=[3, 3], axis=-1, num=2)

    # generate fake image
    y_gen = gen(x, training=True)

    # discriminate real image
    y_pred = dis([x, y], training=True)
    # discriminate fake image
    x_pred = dis([x, y_gen], training=True)

    # compute generator loss
    gen_loss = gen.loss(y, y_gen, x_pred)
    # compute discriminator loss
    dis_loss = dis.loss(y_pred, x_pred)

    return gen_loss, dis_loss


def test(gen, dis, test_set):

    # dataset size
    start = time.time()

    total_gen_loss = []
    total_dis_loss = []
    for step, data in enumerate(test_set):
        # forward one step
        gen_loss, dis_loss = test_step(data, gen, dis)

        total_gen_loss.append(gen_loss.numpy())
        total_dis_loss.append(dis_loss.numpy())

    avg_gen_loss = np.array(total_gen_loss).mean()
    avg_dis_loss = np.array(total_dis_loss).mean()

    end = time.time()
    elapsed_time = datetime.timedelta(seconds=end-start)

    LOG.set_header('Test')
    LOG.add_row('Time elapsed', elapsed_time)
    LOG.add_row('Avg. Generator loss', avg_gen_loss, fmt='{}: {:.6f}')
    LOG.add_row('Avg. Discriminator loss', avg_dis_loss, fmt='{}: {:.6f}')
    LOG.flush('INFO')


# === Training phase ===

@tf.function
def train_step(input, gen, dis, gen_opt, dis_opt):
    # x=input, y=target (real)
    x, y = tf.split(input, num_or_size_splits=[3, 3], axis=3, num=2)

    # since I calling gradient() twice, `persistent` must be set to True
    with tf.GradientTape(persistent=True) as tape:
        # generate fake image
        y_gen = gen(x, training=True)

        # discriminate real image
        y_pred = dis([x, y], training=True)
        # discriminate fake image
        x_pred = dis([x, y_gen], training=True)

        # compute generator loss
        gen_loss = gen.loss(y, y_gen, x_pred)
        # compute discriminator loss
        dis_loss = dis.loss(y_pred, x_pred)
        
    # compute generator gradients
    gen_grads = tape.gradient(gen_loss, gen.trainable_variables)
    # compute discriminator gradients
    dis_grads = tape.gradient(dis_loss, dis.trainable_variables)

    # apply gradients
    gen_opt.apply_gradients(zip(gen_grads, gen.trainable_variables))
    dis_opt.apply_gradients(zip(dis_grads, dis.trainable_variables))

    return gen_loss, dis_loss



def train(gen, dis, gen_opt, dis_opt, checkpoint, train_set, test_set):

    LOG.info('Start training for {} epochs'.format(EPOCHS))

    def plot_sample(data, fname_suffix=''):

        # plotting procedure
        def plot(x, y, y_gen, fname):
    
            plt.figure(figsize=(15, 6))

            def add_subplot(img, title, axis):
                plt.subplot(1, 3, axis, frameon=False)
                plt.title(title, fontsize=24)
                plt.imshow(img * 0.5 + 0.5)
                plt.axis('off')

            add_subplot(x, 'Input Image', 1)
            add_subplot(y_gen, 'Generated Image', 2)
            add_subplot(y, 'Real Image', 3)

            plt.tight_layout()

            # create path and save figure
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            plt.savefig(fname)
            # close figure
            plt.close()
            LOG.info('[Image Saved] save to: {}'.format(fname))

        # if data is a tensor, convert it to numpy array
        if tf.is_tensor(data) and hasattr(data, 'numpy'):
            data = data.numpy()

        # split real/input image
        x, y = np.split(data, 2, axis=-1)
        # reshape images
        x = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        y = y.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        # generate fake image
        y_gen = inference(x, gen, preprocess=False)
        # plot and save image
        plot_path = os.path.join(MODEL_DIR, 'images/{}_epoch_{}.png'.format(MODEL_NAME, fname_suffix))
        plot(x, y, y_gen, plot_path)
            

    # take only one sample for inference
    for train_one_sample in train_set.take(1): pass
    train_one_sample = train_one_sample[0]

    # take only one sample for inference
    for test_one_sample in test_set.take(1): pass
    test_one_sample = test_one_sample[0]

    for epoch in range(EPOCHS):
        start = time.time()

        total_gen_loss = []
        total_dis_loss = []

        for step, data in enumerate(train_set):
            # train for one step
            gen_loss, dis_loss = train_step(data, gen, dis, gen_opt, dis_opt)

            total_gen_loss.append(gen_loss.numpy())
            total_dis_loss.append(dis_loss.numpy())

            if VERBOSE:
                if (step+1) % 100 == 0:
                    # ==== print LOG ====
                    LOG.set_header('Epoch {}/{}'.format(epoch+1, EPOCHS))
                    LOG.add_row('Step', step+1)
                    LOG.add_row('Generator loss', gen_loss, fmt='{}: {:.6f}')
                    LOG.add_row('Discriminator loss', dis_loss, fmt='{}: {:.6f}')
                    LOG.flush('INFO')
                    # ===================

        end = time.time()

        elapsed_time = datetime.timedelta(seconds=end-start)
        avg_gen_loss = np.array(total_gen_loss).mean()
        avg_dis_loss = np.array(total_dis_loss).mean()

        # ==== print LOG ====
        LOG.set_header('Epoch {}/{}'.format(epoch+1, EPOCHS))
        LOG.add_row('Time elapsed', elapsed_time, fmt='{}: {}')
        LOG.add_row('Avg. Generator loss', avg_gen_loss, fmt='{}: {:.6f}')
        LOG.add_row('Avg. Discriminator loss', avg_dis_loss, fmt='{}: {:.6f}')
        LOG.flush('INFO')
        # ====================

        # evaluate model for every EVAL_EPOCHS epochs
        if (EVAL_EPOCHS > 0 and 
                (epoch+1) % EVAL_EPOCHS == 0):
            # evaluate model
            test(gen, dis, test_set)
            # inference one sample
            plot_sample(train_one_sample, '{}_train'.format(epoch+1))
            plot_sample(test_one_sample, '{}_test'.format(epoch+1))

        # save model for every SAVE_EPOCHS epochs
        if (SAVE_EPOCHS > 0 and 
                (epoch+1) % SAVE_EPOCHS == 0):
            # save model
            save_path = os.path.join(MODEL_DIR, MODEL_NAME)
            checkpoint.save(save_path)
            LOG.info('[Model Saved] save to: {}'.format(save_path))

    # evaluate model
    test(gen, dis, test_set)
    # inference one sample
    plot_sample(train_one_sample, 'final_train')
    plot_sample(test_one_sample, 'final_test')


if __name__ == '__main__':

    # download dataset
    data_filename = maybe_download_dataset('facades.tar.gz',
                                       origin='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',
                                       extract=True)
    data_path = os.path.dirname(data_filename)
    

    # create generator
    gen = Generator()
    # create discriminator
    dis = Discriminator()
    # create optimizer for the generator
    gen_opt = tf.optimizers.Adam(learning_rate=GEN_LR, beta_1=0.5)
    # create optimizer for the discriminator
    dis_opt = tf.optimizers.Adam(learning_rate=DIS_LR, beta_1=0.5)

    
    # create checkpoint
    checkpoint = tf.train.Checkpoint(generator=gen,
                                     discriminator=dis,
                                     gen_optimizer=gen_opt,
                                     dis_optimizer=dis_opt)

    gen(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32), training=TRAIN)
    dis(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 6), dtype=np.float32), training=TRAIN)

    # restore checkpoint
    latest_path = tf.train.latest_checkpoint(MODEL_DIR)
    if latest_path is not None:
        LOG.warning('Restore checkpoint from: {}'.format(latest_path))
    checkpoint.restore(latest_path)

    if TRAIN:
        # path = {data_path}/{TRAIN_PATH}/{TRAIN_FILE}
        train_set = create_dataset(path=os.path.join(data_path, os.path.join(TRAIN_PATH, TRAIN_FILE)),
                                is_train=True)
        test_set = create_dataset(path=os.path.join(data_path, os.path.join(TEST_PATH, TEST_FILE)),
                                is_train=False)

        train(gen, dis, gen_opt, dis_opt, checkpoint, train_set, test_set)

    elif INFERENCE:
        
        image = plt.imread(INPUT_PATH)
        output = inference(image, gen, preprocess=True)
        plt.imsave(OUTPUT_PATH, output)

        #plt.imsave(OUTPUT_PATH, output)
        LOG.info('[Image Saved] The image is saved to: {}'.format(OUTPUT_PATH))

    else:
        test_set = create_dataset(path=os.path.join(data_path, os.path.join(TEST_PATH, TEST_FILE)),
                                is_train=False)
        test(gen, dis, test_set)

        
