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

# --- 3rd party ---
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops # get_name_scope() used in auto_naming()

from matplotlib import pyplot as plt

# --- my module ---
sys.path.append('../../')
import logger


# --- Utilities ---

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
    # is not officially documented. (WARNING)
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


# === Primitive Module ===


class Conv(tf.Module):
    def __init__(self, n_kernel,
                       size,
                       stride,
                       gain=0.02,
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

        # --- temps ---
        self.built = False
        self.w = None
        self.b = None
        self._data_format = None
        self._stride = None


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # build module
        if not self.built:

            # get kernel size, stride(height, width)
            k_h, k_w = self.size
            s_h, s_w = self.stride

            # check whether NCHW or NHWC
            if self.channel_first:
                data_format = 'NCHW'
                channel_axis = 1
                stride = [1, 1, s_h, s_w]
            else:
                data_format = 'NHWC'
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

            self._stride = stride
            self._data_format = data_format


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
                                     data_format=self._data_format,
                                     dilations=self.dilations)

        
        if self.is_biased:
            
            # apply bias
            output = tf.nn.bias_add(output, self.b, data_format=self._data_format)

        
        # print logging
        if not self.built:
            
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

            self.built = True

        return output

class Linear(tf.Module):
    def __init__(self, n_hidden,
                       gain=1.0,
                       bias=0.0,
                       data_format=CHANNEL_FIRST,
                       name=None):
        super(Linear, self).__init__(name=auto_naming(name))
        
        self.gain = gain
        self.bias = bias

        self.w_initializer = get_initializer(gain=gain)
        self.b_initializer = get_initializer(bias)

        self.built = False

    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        if not self.built:

            w_shape = []
