# --- built in ---
import os
import sys
import time
import logging
import datetime
import argparse

from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

# --- 3rd party ---
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.utils import tf_utils

# --- my module ---
sys.path.append('../')
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
CHANNEL_FIRST = False  # NCHW (True) or NHWC (False)
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 1  # Training batch size
LOG = None

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
    parser.add_argument('--image_height', type=int, help='The height of images', default=256)
    parser.add_argument('--image_width', type=int, help='The width of images', default=256)
    parser.add_argument('--channel_first', help='Whether to use channel first representation', action='store_true')

    # training parameters
    parser.add_argument('--train', help='Training mode', action='store_false')
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=1)


    # other settings
    parser.add_argument('--seed', type=int, help='Random seed', default=None)
    parser.add_argument('--model_dir', type=str, help='The model directory, default: ./model/ckpt-{timestamp}', 
                                                  default='./model/ckpt-{}'.format(day_timestamp))
    parser.add_argument('--log', type=str, help='The logging path, default: {model_dir}/pix2pix-{timestamp}.log', default=None)
    parser.add_argument('--log_level', type=str, help='The logging level, must be one of [\'DEBUG\', \'INFO\', \'WARNING\']', 
                                                  default=DEFAULT_LOGGING_LEVEL)

    args = parser.parse_args()

    if args.log is None:
        args.log = os.path.join(args.model_dir, 'log/pix2pix-' + sec_timestamp) + '.log'

    return args

def apply_hyperparameters(args):
    global IMAGE_HEIGHT
    global IMAGE_WIDTH
    global BATCH_SIZE
    global LOG

    # assign hyperparameters
    IMAGE_HEIGHT = args.image_height
    IMAGE_WIDTH = args.image_width
    BATCH_SIZE = args.batch_size

    # fixed random seed if specified
    if args.seed is not None:
        tf.random.seed(args.seed)
        np.random.seed(args.seed)

    # create logging path
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    # apply loggin settings
    logger.Config.Use(filename=args.log, level=args.log_level, colored=True, reset=False)
    # create logger
    LOG = logger.getLogger('main')

    # print args
    LOG.set_header('Arguments')

    LOG.subgroup('model')
    LOG.add_row('Image height', args.image_height)
    LOG.add_row('Image width', args.image_width)
    LOG.add_row('Channel first', args.channel_first)

    LOG.subgroup('training')
    LOG.add_row('Training', args.train)
    LOG.add_row('Batch size', args.batch_size)

    LOG.subgroup('others')
    LOG.add_row('Random seed', args.seed)
    LOG.add_row('model directory', args.model_dir)
    LOG.add_row('logging file', args.log)
    LOG.add_row('logging level', args.log_level)

    LOG.flush('INFO')


def verify_settings_2d(s):
    '''
    Used in verifying kernel size and strides
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

    assert padding in {'same', 'valid', 'full'}
    if input_length is None:
        return None

    # dilated kernel size
    size = size + (size - 1) * (dilation - 1)

    if output_padding is None:
        if padding == 'valid':
            length = input_length * stride + max(size - stride, 0)
        elif padding == 'full':
            length = input_length * stride - (stride + size - 2)
        elif padding == 'same':
            length = input_length * stride

    else:
        if padding == 'same':
            pad = size // 2
        elif padding == 'valid':
            pad = 0
        elif padding == 'full':
            pad = size - 1

        length = ((input_length - 1) * stride + size - 2 * pad + output_padding)

    return length


def get_initializer_by_type(*args, type=None, **kwargs):

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
            return get_initializer_by_type(bias)

    if 'seed' in kwargs:
        return tf.initializer.GlorotNormal(seed=seed)

    raise ValueError('Unknown initializer options: {}'.format(kwargs))

        
        

# === Primitive Modules ===

class Conv(tf.Module):
    def __init__(self, n_kernel,
                       size,
                       stride,
                       gain=1.0, 
                       bias=0.0,
                       dilations=1,
                       padding='same',
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

        super(Conv, self).__init__(name=name)

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

            self.has_built = True
            
        
        # perform convolution
        output = tf.nn.conv2d(input, self.w,
                                     strides=stride,
                                     padding=self.padding,
                                     data_format=data_format,
                                     dilations=self.dilations)

        
        if self.is_biased:
            
            # apply bias
            output = tf.nn.bias_add(output, self.b, data_format=self.data_format)

        
        return output



class Deconv(tf.Module):
    def __init__(self, n_kernel,
                       size,
                       stride,
                       gain=1.0,
                       bias=0.0,
                       dilations=1,
                       padding='same',
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

        super(Deconv, self).__init__(name=name)

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


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):
        
        # build module
        if self.has_built:

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
                                               strides=stride,
                                               padding=self.padding,
                                               data_format=data_format,
                                               dilations=self.dilations)

        output.set_shape(self.output_shape)

        if self.is_biased:

            # apply bias
            output = tf.nn.bias_add(output, self.b, data_format=self.data_format)

        
        return output


class BatchNorm(tf.Module):
    def __init__(self, axis=None,
                       momentum=0.99,
                       epsilon=1e-3,
                       gain=1.0,
                       bias=0.0,
                       mean='zeros',
                       var='ones',
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
            mean: (float, str, tf.initializers.Initializer) initial mean
            var: (float, str, tf.initializers.Initializer) initial variance
            channel_first: (bool) NHWC or NCHW
            name: (Optional[str]) module name
        '''
        
        super(BatchNorm, self).__init__(name=name)

        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.gain = gain
        self.bias = bias
        self.mean = mean
        self.var = var
        self.channel_first = True if channel_first else False

        self.gamma_initializer = get_initializer(gain=gain)
        self.beta_initializer = get_initializer(bias=bias)
        self.mean_initializer = get_initializer(mean)
        self.var_initializer = get_initializer(var)

        
        self.has_built = False
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_var = None


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # initialize module
        if not self.is_built:
        
            # get input dimensions
            ndims = len(input.shape[:])
            axis = self.axis

            # verify axis
            if axis is None:
                if self.channel_first:
                    axis = 1
                else:
                    axis = -1
            
            # convert axis to list
            if not isinstance(axis, (list, tuple)):
                axis = [axis]

            # normalize axis (-1 -> ndims-1)
            self.axis = sorted(list(set([ndims+x if x<0 else x for x in axis])))

            # create gamma
            self.gamma = tf.Variable(self.gamma_initializer(self.axis, dtype=tf.float32),
                                     trainable=training,
                                     dtype=tf.float32,
                                     name='gamma')

            # create beta
            self.beta = tf.Variable(self.beta_initializer(self.axis, dtype=tf.float32),
                                    trainable=training,
                                    dtype=tf.float32,
                                    name='beta')

            # create moving mean
            self.moving_mean = tf.Variable(self.mean_initializer(self.axis, dtype=tf.float32),
                                           trainable=False,
                                           dtype=tf.float32,
                                           name='mean')

            # create moving var
            self.moving_var = tf.Variable(self.var_initializer(self.axis, dtype=tf.float32),
                                          trainable=False,
                                          dtype=tf.float32,
                                          name='var')

            self.has_built = True

        def apply_moving_average(variable, value, momentum):
            '''
                a = a * momentum + delta * (1-momentum)
            ->  a = a * (1-decay) + delta * decay   #decay = 1-momentum
            ->  a = a - a * decay + delta * decay
            ->  a = a - (a - delta) * decay
            '''
            decay = tf.convert_to_tensor(1.0 - momentum, dtype=tf.float32, name='decay')

            delta = (variable - value) * decay

            variable.assign_sub(delta=delta, name='apply_moving_average')


        # broadcast gamma/beta into the correct shape
        #   for example, if input shape is [N, 256, 256, 3] in data format `NHWC`,
        #   and perform batch norm on axis `C`, then we have reduction_axes = `NHW`,
        #   broadcast_shape = [1, 1, 1, `C`]
        ndims = len(input.shape[:])
        reduction_axes = [x for x in range(ndims) if x not in self.axis]
        broadcast_shape = [1 if x in reduction_axes else input.shape[x] for x in range(ndims)]

        # correct shapes        
        scale = tf.reshape(self.gamma, broadcast_shape)
        offset = tf.reshape(self.beta, broadcast_shape)

        # compute mean/variance
        mean, var = tf.nn.moments(input, reduction_axes, keepdims=False)

        # update moving mean/variance
        tf_utils.smart_cond(training, 
                            lambda: apply_moving_average(self.moving_mean, mean, self.momentum),
                            lambda: None)
        
        tf_utils.smart_cond(training, 
                            lambda: apply_moving_average(self.moving_var, var, self.momentum),
                            lambda: None)


        # in training mode, using online mean, otherwise, using trained moving_mean
        mean = tf_utils.smart_cond(training,
                                   lambda: mean,
                                   lambda: self.moving_mean)
        
        # in training mode, using online var, otherwise, using trained moving_var
        var = tf_utils.smart_cond(training,
                                  lambda: var,
                                  lambda: self.moving_var, broadcast_shape)

        
        # perform batch normalization
        output = tf.nn.batch_normalization(input,
                                           mean=tf.reshape(mean, broadcast_shape),
                                           variance=tf.reshape(var, broadcast_shape),
                                           offset=offset,
                                           scale=scale,
                                           variance_epsilon=self.epsilon)

        output.set_shape(input.shape)

        return output


class Dropout(tf.Module):
    def __init__(self, rate,
                       noise_shape=None,
                       seed=None,
                       name=None):

        super(Dropout, self).__init__(name=name)

        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        output = tf_utils.smat_cond(trianing,
                                    lambda: tf.nn.dropout(input,
                                                          noise_shape=self.noise_shape,
                                                          seed=self.seed),
                                    lambda: tf.identity(input))
        
        return output
    

class Padding(tf.Module):
    def __init__(self, padding=1,
                       mode='CONSTANT', 
                       constant_values=0,
                       channel_first=CHANNEL_FIRST,
                       name=None):
        
        super(Padding, self).__init__(name=name)

        self.padding = verify_settings_2d(padding)
        self.mode = mode
        self.constant_values = constant_values
        self.channel_first = channel_first

        self.has_built = False
        self.pad_shape = None

    def __call__(self, input, training=True):

        # build module
        if not self.has_built:

            ndims = len(input.shape[:])
            assert ndims == 4

            p_h, p_w = self.padding

            # find height/width axis
            if self.channel_first: # NCHW
                height_axis, width_axis = 2, 3
            else: # NHWC
                height_axis, width_axis = 1, 2
                
            # create padding shape
            self.pad_shape = [ [0, 0] for d in range(ndims) ]
            self.pad_shape[height_axis] = [p_h, p_h]  # pad height
            self.pad_shape[width_axis] = [p_w, p_w]  # pad width


            self.has_built = True

        output = tf.pad(input, self.pad_shape, 
                               mode=self.mode, 
                               constant_values=self.constant_values)
        
        return output



# === Sub Modules ===

class DownSample(tf.Module):
    def __init__(self, n_kernel,
                       size,
                       stride=2,
                       apply_batchnorm=True,
                       apply_activ=True,
                       name=None):

        super(DownSample, self).__init__(name=name)

        self.apply_batchnorm = apply_batchnorm
        self.apply_activ = apply_activ

        # create convolution layer
        self.conv = Conv(n_kernel=n_kernel,
                         size=size,
                         stride=stride,
                         padding='same',
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
                       apply_batchnorm=True,
                       apply_activ=True,
                       apply_dropout=False,
                       name=None):

        super(UpSample, self).__init__(name=name)

        self.apply_batchnorm = apply_batchnorm
        self.apply_activ = apply_activ
        self.apply_dropout = apply_dropout

        # create deconvolution layer
        self.deconv = Deconv(n_kernel=n_kernel,
                           size=size,
                           stride=2,
                           padding='same',
                           is_biased=False)

        # create batch normalization layer
        if self.apply_batchnorm:
            self.batchnorm = BatchNorm()

        # create dropout layer
        if self.apply_dropout:
            self.dropout = Dropout(0.5)


    def __call__(self, input, training=True):
        
        # concatenate inputs
        if isinstance(input, (list, tuple)):
            input = tf.concat(input, axis=-1)

        # forward deconvolution
        output = self.deconv(input, training=training)
        
        # forward batch normalization
        if self.apply_batchnorm:
            output = self.batchnorm(input, training=training)

        # apply dropout
        if self.apply_dropout:
            output = self.dropout(output, training=training)

        # apply relu
        if self.apply_activ:
            output = tf.nn.relu(output)

        return output



class UNet(tf.Module):
    def __init__(self, name=None):
        super(UNet, self).__init__(name=name)

        self.downsample_spec = [dict(n_kernel=64,  size=4, apply_batchnorm=False),
                                dict(n_kernel=128, size=4),
                                dict(n_kernel=256, size=4),
                                dict(n_kernel=512, size=4),
                                dict(n_kernel=512, size=4),
                                dict(n_kernel=512, size=4),
                                dict(n_kernel=512, size=4),
                                dict(n_kernel=512, size=4) ]

        self.upsample_spec = [dict(n_kernel=512, size=4, apply_dropout=True),
                              dict(n_kernel=512, size=4, apply_dropout=True),
                              dict(n_kernel=512, size=4, apply_dropout=True),
                              dict(n_kernel=512, size=4),
                              dict(n_kernel=512, size=4),
                              dict(n_kernel=512, size=4),
                              dict(n_kernel=512, size=4),
                              dict(n_kernel=3, size=4, apply_batchnorm=False, apply_activ=False)]

        self.downsamples = [ DownSample(**spec) for spec in self.downsample_spec ]
        self.upsamples = [ UpSample(**spec) for spec in self.upsample_spec ]

    
    def __call__(self, input, training=None):

        outputs = []

        # UNet down sampling
        for down in self.downsamples:
            # forward downsample layers
            output = down(input, training=training)
            # append forwarded results
            outputs.append(output)
            # prepare inputs for the next layer
            input = output

        # remove the last output
        outputs = reversed(outputs)
        
        # UNet up sampling
        for up, out in zip(self.upsamples, outputs):
            # for the first upsample layer, the input is the output from the 
            # last downsample layer. Hence, only pass `input` as layer inputs.
            # otherwise, pass `input` with `out` as layer inputs.
            if out is input:
                output = up(input, training=training)
            else:
                output = up([input, out], training=training)

            # prepare inputs for the next layer
            input = output

        # final activation function, normalize output between [-1, 1]
        output = tf.nn.tanh(input)

        return output


# === Main Modules ===

class Generator(UNet):
    def __init__(self, name='Generator'):
        super(Generator, self).__init__(name=name)

class Discriminator(tf.Module):
    def __init__(self, name='Discriminator'):
        super(Discriminator, self).__init__(name=name)

        self.down1 = DownSample(64, 4, apply_batchnorm=False)
        self.down2 = DownSample(128, 4)
        self.down3 = DownSample(256, 4)
        
        self.pad1 = Padding()
        self.down4 = DownSample(512, 4, stride=1)

        self.pad2 = Padding()
        self.conv2 = Conv(1, 4, stride=1, is_biased=True)

    def __call__(self, input, training=True):

        # concatenate multiple inputs
        if isinstance(input, (tuple, list)):
            input = tf.concat(input, axis=-1)

        output = self.down1(input, training=training)
        output = self.down2(output, training=training)
        output = self.down3(output, training=training)

        output = self.pad1(output, training=training)
        output = self.down4(output, training=training)

        output = self.pad2(output, training=training)
        output = self.conv2(output, training=training)

        return output




if __name__ == '__main__':
    apply_hyperparameters(parse_args())