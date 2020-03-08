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


'''
TensorFlow 2.0 implementation of CycleGAN

The implementation follows the original paper version of CycleGAN, introduced by Jun-Yan Zhu et al.

Reference:
    The original paper: Unpaired Image-to-Image Translation using Cycle-Consistent Adversatial Networks
    Arxiv: https://arxiv.org/abs/1703.10593
    Github: https://github.com/junyanz/CycleGAN
'''

# === GPU settings ===
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# === Tensorflow settings ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# === Hyper-parameters ===
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNEL_FIRST = False  # True => NCHW, False => NHWC

INFERENCE = False   # inference mode
INPUT_PATH = None   # input image path
OUTPUT_PATH = None  # output image path

TRAIN = False          # train or test mode
BATCH_SIZE = 1         # training batch size
CYCLE_A_LAMBDA = 10.0  # forward cycle consistency loss (A -> B -> A)
CYCLE_B_LAMBDA = 10.0  # backward cycle consistency loss (B -> A -> B)
ID_LAMBDA = 0.5        # identity loss
GEN_LR = 2e-4          # generator learning rate
DIS_LR = 2e-4          # discriminator learning rate
EPOCHS = 1000          # total training epochs
EVAL_EPOCHS = 1        # evalute per epochs
SAVE_EPOCHS = 10       # save checkpoint for every 10 epochs
#EXPORT_BEST = False
EXPORT_LATEST = True   # export latest model

DATA_FILENAME = 'horse2zebra.zip'
DATA_URL = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip'
DATA_ROOT = 'datasets'
TRAIN_A_PATH = 'horse2zebra/trainA'
TRAIN_B_PATH = 'horse2zebra/trainB'
TEST_A_PATH = 'horse2zebra/testA'
TEST_B_PATH = 'horse2zebra/testB'
TRAIN_A_FILE = '*.jpg'      # {TRAIN_A_PATH}/{TRAIN_A_FILE}: horse2zebra/trainA/*.jpg
TRAIN_B_FILE = '*.jpg'
TEST_A_FILE = '*.jpg'
TEST_B_FILE = '*.jpg'
LABELS = ['Horse', 'Zebra', 'Horse -> Zebra', 'Zebra -> Horse']

SEED=None          # random seed
MODEL_DIR = None   # checkpoint path
MODEL_NAME = None  # checkpoint name
LOG_PATH = None    # logging path
LOG_LEVEL = 'INFO' # logging level
LOG = None         # logger
VERBOSE = False    # show more log info

BUFFER_SIZE = 100  # buffer size for shuffling training datasets

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
    parser.add_argument('--lambda1', dest='cycle_a_lambda', type=float, help='A factor for the forward-pass (A->B->A) cycle consistency loss', default=CYCLE_A_LAMBDA)
    parser.add_argument('--lambda2', dest='cycle_b_lambda', type=float, help='A factor for the backward-pass (B->A->B) cycle consistency loss', default=CYCLE_B_LAMBDA)
    parser.add_argument('--lambda3', dest='id_lambda', type=float, help='A factor for the generator identity loss', default=ID_LAMBDA)
    parser.add_argument('--gen_lr', type=float, help='The learning rate of Adam for the generator', default=GEN_LR)
    parser.add_argument('--dis_lr', type=float, help='The learning rate of Adam for the discriminator', default=DIS_LR)
    parser.add_argument('--epochs', type=int, help='Training epochs', default=EPOCHS)
    parser.add_argument('--eval_epochs', type=int, help='Evaluate every N epochs', default=EVAL_EPOCHS)
    parser.add_argument('--save_epochs', type=int, help='Save model for every N epochs', default=SAVE_EPOCHS)
    #parser.add_argument('--export_best', help='Whether to export the best model', action='store_true')

    # dataset parameters
    parser.add_argument('--data_filename', type=str, help='The filename of the dataset', default=DATA_FILENAME)
    parser.add_argument('--data_url', type=str, help='The url of the dataset', default=DATA_URL)
    parser.add_argument('--data_root', type=str, help='The root path to store the downloaded datasets', default=DATA_ROOT)
    parser.add_argument('--train_a_path', type=str, help='The root path to the training set of pair A', default=TRAIN_A_PATH)
    parser.add_argument('--train_b_path', type=str, help='The root path to the training set of pair B', default=TRAIN_B_PATH)
    parser.add_argument('--test_a_path', type=str, help='The root path to the testing set of pair A', default=TEST_A_PATH)
    parser.add_argument('--test_b_path', type=str, help='The root path to the testing set of pair B', default=TEST_B_PATH)
    parser.add_argument('--train_a_file', type=str, help='The filename of the training set of pair A', default=TRAIN_A_FILE)
    parser.add_argument('--train_b_file', type=str, help='The filename of the training set of pair B', default=TRAIN_B_FILE)
    parser.add_argument('--test_a_file', type=str, help='The filename of the test set of pair A', default=TEST_A_FILE)
    parser.add_argument('--test_b_file', type=str, help='The filename of the test set of pair B', default=TEST_B_FILE)

    # other settings
    parser.add_argument('--seed', type=int, help='Random seed', default=None)
    parser.add_argument('--model_dir', type=str, help='The model directory, default: ./model/ckpt-{timestamp}', 
                                                  default='./model/ckpt-{}'.format(day_timestamp))
    parser.add_argument('--model_name', type=str, help='The name of the model', default='model')
    parser.add_argument('--log_path', type=str, help='The logging path, default: {model_dir}/pix2pix-{timestamp}.log', default=LOG_PATH)
    parser.add_argument('--log_level', type=str, help='The logging level, must be one of [\'DEBUG\', \'INFO\', \'WARNING\']', 
                                                  default=LOG_LEVEL)
    parser.add_argument('-v', '--verbose', help='If True, more loggin is printed', action='store_true')

    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = os.path.join(args.model_dir, 'log/cyclegan-' + sec_timestamp) + '.log'

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
    global CYCLE_A_LAMBDA
    global CYCLE_B_LAMBDA
    global ID_LAMBDA
    global GEN_LR
    global DIS_LR
    global EPOCHS
    global EVAL_EPOCHS
    global SAVE_EPOCHS
    #global EXPORT_BEST
    global EXPORT_LATEST

    global DATA_FILENAME
    global DATA_URL
    global DATA_ROOT
    global TRAIN_A_PATH
    global TRAIN_B_PATH
    global TEST_A_PATH
    global TEST_B_PATH
    global TRAIN_A_FILE
    global TRAIN_B_FILE
    global TEST_A_FILE
    global TEST_B_FILE

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
    CYCLE_A_LAMBDA = args.cycle_a_lambda
    CYCLE_B_LAMBDA = args.cycle_b_lambda
    ID_LAMBDA = args.id_lambda
    GEN_LR = args.gen_lr
    DIS_LR = args.dis_lr
    EPOCHS = args.epochs
    EVAL_EPOCHS = args.eval_epochs
    SAVE_EPOCHS = args.save_epochs
    #EXPORT_BEST = args.export_best
    #EXPORT_LATEST = not args.export_best

    DATA_FILENAME = args.data_filename
    DATA_URL = args.data_url
    DATA_ROOT = args.data_root
    TRAIN_A_PATH = args.train_a_path
    TRAIN_B_PATH = args.train_b_path
    TEST_A_PATH = args.test_a_path
    TEST_B_PATH = args.test_b_path
    TRAIN_A_FILE = args.train_a_file
    TRAIN_B_FILE = args.train_b_file
    TEST_A_FILE = args.test_a_file
    TEST_B_FILE = args.test_b_file

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
    LOG.add_row()
    LOG.add_rows('Cycle GAN', fmt='{:@f:ANSI_Shadow}', align='center')
    LOG.add_row('Paper: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks')
    LOG.add_row('Author: Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A')
    LOG.add_line()
    
    LOG.add_row('TensorFlow 2.0 implementation.')
    LOG.add_row('Implemented by Tsu-Ching Hsiao (Ending2015a) on 2020.03.07', align='right')
    LOG.add_row('Github: https://github.com/Ending2015a')
    LOG.add_row('Project: https://github.com/Ending2015a/allintf2')
    
    LOG.flush('INFO')

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
        LOG.add_row('Cycle A lambda', CYCLE_A_LAMBDA)
        LOG.add_row('Cycle B lambda', CYCLE_B_LAMBDA)
        LOG.add_row('Identity lambda', ID_LAMBDA)
        LOG.add_row('Generator lr', GEN_LR)
        LOG.add_row('Discriminator lr', DIS_LR)
        LOG.add_row('Eval epochs', EVAL_EPOCHS)
        LOG.add_row('Save epochs', SAVE_EPOCHS)
        #LOG.add_row('Export best', EXPORT_BEST)
        LOG.add_row('Export latest', EXPORT_LATEST)
    
    if not INFERENCE:
        # training or testing mode
        LOG.subgroup('dataset')
        LOG.add_row('Data filename', DATA_FILENAME)
        LOG.add_row('Data URL', DATA_URL)
        LOG.add_row('Data root', DATA_ROOT)
        LOG.add_row('Training set A path', os.path.join(TRAIN_A_PATH, TRAIN_A_FILE))
        LOG.add_row('Training set B path', os.path.join(TRAIN_B_PATH, TRAIN_B_FILE))
        LOG.add_row('Test set A path', os.path.join(TEST_A_PATH, TEST_A_FILE))
        LOG.add_row('Test set B path', os.path.join(TEST_B_PATH, TEST_B_FILE))

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


# ====== Main ======
if __name__ == '__main__':
    # set hyperparameters
    apply_hyperparameters(parse_args())
# ==================


# === Utility ===

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

        self.built = False
        self.w = None
        self.b = None
        self._data_format = None
        self._output_shape = None
        self._stride = None


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):
        
        # build module
        if not self.built:

            # get kernel size, stride, dilations, output padding (height, width)
            k_h, k_w = self.size
            s_h, s_w = self.stride
            d_h, d_w = self.dilations
            p_h, p_w = self.output_padding

            # check whether NCHW or NHWC
            if self.channel_first:
                data_format = 'NCHW'
                channel_axis, height_axis, width_axis = 1, 2, 3
                stride = [1, 1, s_h, s_w]
            else:
                data_format = 'NHWC'
                height_axis, width_axis, channel_axis = 1, 2, 3
                stride = [1, s_h, s_w, 1]

            self._stride = stride
            self._data_format = data_format

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
                output_shape = [input.shape[0], c_o, o_h, o_w]
            else:
                output_shape = [input.shape[0], o_h, o_w, c_o]

            self._output_shape = output_shape


        # perform deconvolution
        output = tf.nn.conv2d_transpose(input, self.w,
                                               output_shape=tf.convert_to_tensor(self._output_shape),
                                               strides=self._stride,
                                               padding=self.padding,
                                               data_format=self._data_format,
                                               dilations=self.dilations)

        output.set_shape(self._output_shape)


        if self.is_biased:

            # apply bias
            output = tf.nn.bias_add(output, self.b, data_format=self._data_format)


        # print logging
        if not self.built:
            
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

            self.built = True
        
        return output


class BatchNorm(tf.Module):
    def __init__(self, axis=None,
                       momentum=0.99,
                       epsilon=1e-3,
                       gain='ones',
                       bias='zeros',
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

        
        self.built = False
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_var = None
        self.broadcast_shape = None
        self.reduction_axes = None
        self._data_format = None

        # set default axis if axis is None
        if self.axis is None:
            self.axis = 1 if self.channel_first else -1

        # convert to list
        if not isinstance(self.axis, (list, tuple)):
            self.axis = [self.axis]



    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # initialize module
        if not self.built:
        
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

            # decide data format
            self._data_format = 'NCHW' if self.channel_first else 'NHWC'


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
                                    data_format=self._data_format)
            

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
                                    data_format=self._data_format,
                                    is_training=False)
            


        if not self.built:

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

            self.built = True

        return output



class InstanceNorm(tf.Module):
    def __init__(self, epsilon=1e-5,
                       gain='ones',
                       bias='zeros',
                       channel_first=CHANNEL_FIRST,
                       name=None):
        super(InstanceNorm, self).__init__(name=auto_naming(self, name))

        self.epsilon = epsilon
        self.gain = gain
        self.bias = bias
        self.channel_first = channel_first

        self.gamma_initializer = get_initializer(gain)
        self.beta_initializer = get_initializer(bias)

        self.built = False
        self.reduction_axes = None
        self.broadcast_shape = None
        self.gamma = None
        self.beta = None
        

    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        if not self.built:
            
            ndims = len(input.shape)

            self.broadcast_shape = [1] * ndims

            if ndims > 2:
                if self.channel_first:
                    channel_axis = 1
                    self.reduction_axes = [x for x in range(2, ndims)]
                else:
                    channel_axis = -1
                    self.reduction_axes = [x for x in range(1, ndims-1)]
            else:
                channel_axis = -1
                self.reduction_axes = [ndims-1]
            

            shape = input.shape[channel_axis]
            
            self.broadcast_shape = [1] * ndims
            self.broadcast_shape[channel_axis] = shape


            self.gamma = tf.Variable(self.gamma_initializer(shape, dtype=tf.float32),
                                     trainable=training,
                                     dtype=tf.float32,
                                     name='gamma')

            self.beta = tf.Variable(self.beta_initializer(shape, dtype=tf.float32),
                                    trainable=training,
                                    dtype=tf.float32,
                                    name='beta')
        
        offset = tf.reshape(self.beta, self.broadcast_shape)
        scale = tf.reshape(self.gamma, self.broadcast_shape)

        # compute mean, variance
        mean, var = tf.nn.moments(input, axes=self.reduction_axes, keepdims=True)
        
        

        # apply instance normalization
        output = tf.nn.batch_normalization(input, mean=mean,
                                           variance=var,
                                           offset=offset,
                                           scale=scale,
                                           variance_epsilon=self.epsilon)

        if not self.built:

            LOG.set_header('InstanceNorm \'{}\''.format(self.name))
            
            LOG.add_row('scope', '{}'.format(self.name_scope.name))
            LOG.add_row('training', training)

            LOG.subgroup('shape')
            LOG.add_row('input shape', input.shape)
            LOG.add_row('output shape', output.shape)

            if VERBOSE:
                LOG.subgroup('config')
                LOG.add_row('eps', self.epsilon)
                LOG.add_row('gain', self.gain)
                LOG.add_row('bias', self.bias)
                LOG.add_row('channel first', self.channel_first)
                LOG.add_row('reduction axes', self.reduction_axes)
                LOG.add_row('broadcast_shape', self.broadcast_shape)

            LOG.flush('DEBUG')

            self.built = True

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

        self.built = False


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):



        # if training, use user specified dropout rate, otherwise ignore dropout (rate=0.0)
        rate = self.rate if training else 0.0

        output = tf.nn.dropout(input, rate=rate,
                                      noise_shape=self.noise_shape,
                                      seed=self.seed)

        # print logging
        if not self.built:
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

            self.built = True

        
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

        self.built = False
        self.pad_shape = None


    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # build module
        if not self.built:

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
        if not self.built:
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

            self.built = True
        
        return output

# === Modules ===

class DownSample(tf.Module):
    def __init__(self, n_kernel,
                       size,
                       stride=2,
                       apply_norm=True,
                       apply_activ=True,
                       norm_type=InstanceNorm,
                       name=None):
        super(DownSample, self).__init__(name=auto_naming(self, name))

        self.apply_norm = apply_norm
        self.apply_activ = apply_activ

        self.conv = Conv(n_kernel=n_kernel, 
                         size=size, 
                         stride=stride,
                         padding='SAME',
                         is_biased=False)
        
        if self.apply_norm:
            self.norm = norm_type()
        

    def __call__(self, input, training=True):

        output = self.conv(input, training=training)

        if self.apply_norm:
            output = self.norm(output, training=training)

        if self.apply_activ:
            output = tf.nn.leaky_relu(output, alpha=0.2)

        return output

class UpSample(tf.Module):
    def __init__(self, n_kernel, 
                       size,
                       stride=2,
                       apply_norm=True,
                       apply_dropout=False,
                       apply_activ=True,
                       norm_type=InstanceNorm,
                       name=None):

        super(UpSample, self).__init__(name=auto_naming(self, name))

        self.apply_norm = apply_norm
        self.apply_activ = apply_activ
        self.apply_dropout = apply_dropout

        # create deconvolution layer
        self.deconv = Deconv(n_kernel=n_kernel,
                           size=size,
                           stride=stride,
                           padding='SAME',
                           is_biased=False)

        if self.apply_norm:
            self.norm = norm_type()
        
        if self.apply_dropout:
            self.dropout = Dropout(0.5)

    def __call__(self, input, training=True):

        output = self.deconv(input, training=training)

        if self.apply_norm:
            output = self.norm(output, training=training)

        if self.apply_dropout:
            output = self.dropout(output, training=training)

        if self.apply_activ:
            output = tf.nn.relu(output)

        return output

# === Generator ===

class EncoderDecoder(tf.Module):
    '''
    Implementation of the encoder decoder structure

    Reference: 
        https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L56-L95
    '''
    def __init__(self, name=None):
        super(EncoderDecoder, self).__init__(name=auto_naming(self, name))

        # layer definitions
        # in default ngf=64
        # input shape: [batch, 256, 256, 3]
        self.downsample_spec = [dict(n_kernel=64,  size=4, apply_norm=False),      # output shape: [batch, 128, 128, ngf]
                                dict(n_kernel=128, size=4),                        # output shape: [batch, 64, 64, ngf*2]
                                dict(n_kernel=256, size=4),                        # output shape: [batch, 32, 32, ngf*4]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 16, 16, ngf*8]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 8, 8, ngf*8]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 4, 4, ngf*8]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 2, 2, ngf*8]
                                dict(n_kernel=512, size=4, apply_norm=False, apply_activ=False) ] # output shape: [batch, 1, 1, ngf*8]
        
        self.upsample_spec = [dict(n_kernel=512, size=4, apply_dropout=True),      # output shape: [batch, 2, 2, ngf*8]
                              dict(n_kernel=512, size=4, apply_dropout=True),      # output shape: [batch, 4, 4, ngf*8]
                              dict(n_kernel=512, size=4, apply_dropout=True),      # output shape: [batch, 8, 8, ngf*8]
                              dict(n_kernel=512, size=4),                          # output shape: [batch, 16, 16, ngf*8]
                              dict(n_kernel=256, size=4),                          # output shape: [batch, 32, 32, ngf*4]
                              dict(n_kernel=128, size=4),                          # output shape: [batch, 64, 64, ngf*2]
                              dict(n_kernel=64, size=4),                           # output shape: [batch, 128, 128, ngf]
                              dict(n_kernel=3, size=4, apply_norm=False, apply_activ=False)] # output shape: [batch, 256, 256, 3]

        # create layers
        with self.name_scope:
            self.downsamples = [DownSample(**spec) for spec in self.downsample_spec]
            self.upsamples = [UpSample(**spec) for spec in self.upsample_spec]

    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # input size = 256 x 256

        # encoder down sampling
        for layer in self.downsamples:
            # forward downsample layers  
            output = layer(input, training=training)
            # prepare inputs for the next layer
            input = output

        # DEBUG: check bottle neck
        #LOG.debug('training = {}, scope = {}'.format(training, self.downsamples[-1].name_scope.name))
        #LOG.debug('output shape = {}, output = \n{}'.format(input.shape, input.numpy()[0, :, :, 100])) #1*1*100

        input = tf.nn.relu(input)

        for layer in self.upsamples:
            # forward upsample layers
            output = layer(input, training=training)
            # prepare inputs for the next layer
            input = output

        # DEBUG: check network outputs
        #LOG.debug('training = {}, scope = {}'.format(training, self.upsamples[-1].name_scope.name))
        #LOG.debug('output shape = {}, output = \n{}'.format(input.shape, input.numpy()[0, :3, :10, :])) #3*10*3

        # final activation function, normalize output between [-1, 1]
        output = tf.nn.tanh(input)

        return output

class UNet128(tf.Module):
    '''
    Implementation of the unet128 network

    Reference:
        https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L98-L141
    '''
    def __init__(self, name=None):
        super(UNet128, self).__init__(name=auto_naming(self, name))

        # layer definitions
        # in default ngf=64
        # input shape: [batch, 128, 128, 3]
        self.downsample_spec = [dict(n_kernel=64,  size=4, apply_norm=False),      # output shape: [batch, 64, 64, ngf]
                                dict(n_kernel=128, size=4),                        # output shape: [batch, 32, 32, ngf*2]
                                dict(n_kernel=256, size=4),                        # output shape: [batch, 16, 16, ngf*4]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 8, 8, ngf*8]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 4, 4, ngf*8]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 2, 2, ngf*8]
                                dict(n_kernel=512, size=4, apply_norm=False, apply_activ=False) ] # output shape: [batch, 1, 1, ngf*8]
        
        self.upsample_spec = [dict(n_kernel=512, size=4, apply_dropout=True),      # output shape: [batch, 2, 2, ngf*8]
                              dict(n_kernel=512, size=4, apply_dropout=True),      # output shape: [batch, 4, 4, ngf*8]
                              dict(n_kernel=512, size=4, apply_dropout=True),      # output shape: [batch, 8, 8, ngf*8]
                              dict(n_kernel=256, size=4),                          # output shape: [batch, 16, 16, ngf*4]
                              dict(n_kernel=128, size=4),                          # output shape: [batch, 32, 32, ngf*2]
                              dict(n_kernel=64, size=4),                           # output shape: [batch, 64, 64, ngf]
                              dict(n_kernel=3, size=4, apply_norm=False, apply_activ=False)] # output shape: [batch, 128, 128, 3]

        with self.name_scope:
            self.downsamples = [DownSample(**spec) for spec in self.downsample_spec]
            self.upsamples = [UpSample(**spec) for spec in self.upsample_spec]

    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # input size = 128 x 128
        outputs = []

        # UNet down sampling
        for layer in self.downsamples:
            # forward downsample layers  
            output = layer(input, training=training)
            # append forwarded results to the list
            outputs.append(output)
            # prepare inputs for the next layer
            input = output

        # DEBUG: check bottle neck
        #LOG.debug('training = {}, scope = {}'.format(training, self.downsamples[-1].name_scope.name))
        #LOG.debug('output shape = {}, output = \n{}'.format(input.shape, input.numpy()[0, :, :, 100])) #1*1*100

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
        
        # DEBUG: check network outputs
        #LOG.debug('training = {}, scope = {}'.format(training, self.upsamples[-1].name_scope.name))
        #LOG.debug('output shape = {}, output = \n{}'.format(input.shape, input.numpy()[0, :3, :10, :])) #3*10*3

        # final activation function, normalize output between [-1, 1]
        output = tf.nn.tanh(input)

        return output

class UNet256(tf.Module):
    '''
    Implmementation of the unet256 network

    Reference:
        https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L144-L191
    '''
    def __init__(self, name=None):
        super(UNet256, self).__init__(name=auto_naming(self, name))

        # layer definitions
        # in default ngf=64
        # input shape: [batch, 256, 256, 3]
        self.downsample_spec = [dict(n_kernel=64,  size=4, apply_norm=False),      # output shape: [batch, 128, 128, ngf]
                                dict(n_kernel=128, size=4),                        # output shape: [batch, 64, 64, ngf*2]
                                dict(n_kernel=256, size=4),                        # output shape: [batch, 32, 32, ngf*4]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 16, 16, ngf*8]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 8, 8, ngf*8]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 4, 4, ngf*8]
                                dict(n_kernel=512, size=4),                        # output shape: [batch, 2, 1, ngf*8]
                                dict(n_kernel=512, size=4, apply_norm=False, apply_activ=False) ] # output shape: [batch, 1, 1, ngf*8]
        
        self.upsample_spec = [dict(n_kernel=512, size=4, apply_dropout=True),      # output shape: [batch, 2, 2, ngf*8]
                              dict(n_kernel=512, size=4, apply_dropout=True),      # output shape: [batch, 4, 4, ngf*8]
                              dict(n_kernel=512, size=4, apply_dropout=True),      # output shape: [batch, 8, 8, ngf*8]
                              dict(n_kernel=512, size=4),                          # output shape: [batch, 16, 16, ngf*8]
                              dict(n_kernel=256, size=4),                          # output shape: [batch, 32, 32, ngf*4]
                              dict(n_kernel=128, size=4),                          # output shape: [batch, 64, 64, ngf*2]
                              dict(n_kernel=64, size=4),                           # output shape: [batch, 128, 128, ngf]
                              dict(n_kernel=3, size=4, apply_norm=False, apply_activ=False)] # output shape: [batch, 256, 256, 3]

        # create layers
        with self.name_scope:
            self.downsamples = [DownSample(**spec) for spec in self.downsample_spec]
            self.upsamples = [UpSample(**spec) for spec in self.upsample_spec]

    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        # input size = 256 x 256
        outputs = []

        # UNet down sampling
        for layer in self.downsamples:
            # forward downsample layers  
            output = layer(input, training=training)
            # append forwarded results to the list
            outputs.append(output)
            # prepare inputs for the next layer
            input = output

        # DEBUG: check bottle neck
        #LOG.debug('training = {}, scope = {}'.format(training, self.downsamples[-1].name_scope.name))
        #LOG.debug('output shape = {}, output = \n{}'.format(input.shape, input.numpy()[0, :, :, :100])) #1*1*100

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

        # DEBUG: check network outputs
        #LOG.debug('training = {}, scope = {}'.format(training, self.upsamples[-1].name_scope.name))
        #LOG.debug('output shape = {}, output = \n{}'.format(input.shape, input.numpy()[0, :1, :10, :])) #1*10*3

        # final activation function, normalize output between [-1, 1]
        output = tf.nn.tanh(input)

        return output


class ConvBlock(tf.Module):
    '''
    Implementation of the conv block structure

    Reference:
        https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L197-L218

    TODO:
        Different type of padding: 'reflect', 'replicate'
    '''
    def __init__(self, n_kernel,
                       size=3,
                       stride=1,
                       norm_type=InstanceNorm,
                       name=None):
        super(ConvBlock, self).__init__(name=auto_naming(self, name))

        self.conv1 = Conv(n_kernel=n_kernel, 
                          size=size, 
                          stride=stride,
                          padding='SAME',
                          is_biased=False)

        self.norm1 = norm_type()
        self.conv2 = Conv(n_kernel=n_kernel,
                          size=size,
                          stride=stride,
                          padding='SAME',
                          is_biased=False)

        self.norm2 = norm_type()

    def __call__(self, input, training=True):

        output = self.conv1(input, training=training)
        output = self.norm(output, training=training)
        output = tf.nn.relu(output)
        output = self.conv2(output, training=training)
        output = self.norm(output, training=training)

        return output

class ResBlock(tf.Module):
    '''
    Implementation of the resnet block structure

    Reference:
        https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L221-L230
    '''
    def __init__(self, n_kernel,
                       norm_type=InstanceNorm,
                       name=None):

        super(ResBlock, self).__init__(name=auto_naming(self, name))

        with self.name_scope:
            self.conv_block = ConvBlock(n_kernel, norm_type=norm_type)

    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        output = self.conv_block(input, training=training)

        output = tf.math.add(output, input)

        return output



class Generator(tf.Module):
    def __init__(self, net=UNet256, loss_type='BCE', name=None):
        super(Generator, self).__init__(name=auto_naming(self, name))

        self.loss_type = loss_type

        # create sub-modules under the name scope
        with self.name_scope:
            self.net = net()

    @tf.function
    @tf.Module.with_name_scope
    def __call__(self, input, training=True):

        output = self.net(input, training=training)

        return output

    def _abs_loss(self, y_pred, x_pred):
        '''
        Discriminator mean square error loss

        Args:
            y_pred: D(y)
            x_pred: D(G(x))

        Reference:
            https://github.com/junyanz/CycleGAN/blob/master/models/cycle_gan_model.lua#L136-L153
        '''

        real_loss = tf.math.reduce_mean(
                        tf.math.abs(tf.ones_like(y_pred) - y_pred))

        gened_loss = tf.math.reduce_mean(
                        tf.math.abs(tf.zeros_like(x_pred) - x_pred))

        loss = (real_loss + gened_loss) * 0.5
        return loss


    def loss(self, x, y, x_regen, y_regen, x_pred, y_id, lambda1, lambda2, verbose=False):
        '''
        generator loss

        assume generator G: x -> y, F: y -> x

        L = L_gan + lambda1 * L_fw_cyc + lambda2 * L_bw_cyc + id_lambda * L_id

        L_gan = -E[log(D(G(x)))]
        L_fw_cyc = E[|F(G(x)) - x|]
        L_bw_cyc = E[|G(F(y)) - y|]
        L_id = E[|G(y) - y|]

        Args:
            x: x
            y: y
            x_regen: F(G(x))
            y_regen: G(F(y))
            x_pred: D(G(x))
            y_id: G(y)
            

        Reference:
            https://developers.google.com/machine-learning/gan/loss#modified-minimax-loss
        '''

        # --- GAN loss ---
        # Reference:
        #     https://github.com/junyanz/CycleGAN/blob/master/models/cycle_gan_model.lua#L190-L195
        if self.loss_type == 'BCE':
            gan_loss = tf.math.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=tf.ones_like(x_pred), logits=x_pred))

        elif self.loss_type == 'MSE':
            gan_loss = tf.math.reduce_mean(
                            tf.math.squared_difference(
                                tf.ones_like(x_pred), x_pred))
        else:
            raise NotImplementedError('The method for loss_type \'{}\' is not implemented'.format(self.loss_type))

        # --- Forward cycle consistency loss ---
        # Reference:
        #     https://github.com/junyanz/CycleGAN/blob/master/models/cycle_gan_model.lua#L197-L201
        fw_loss = tf.math.reduce_mean(tf.math.abs(x_regen - x))

        # --- Backward cycle consistency loss ---
        # Reference:
        #     https://github.com/junyanz/CycleGAN/blob/master/models/cycle_gan_model.lua#L205-L210
        bw_loss = tf.math.reduce_mean(tf.math.abs(y_regen - y))

        # --- Identity loss ---
        # Described in the Sec.5.2 "Photo generation from paintings (Figure 12) " in CycleGan paper
        # Reference:
        #     https://github.com/junyanz/CycleGAN/blob/master/models/cycle_gan_model.lua#L180-L188
        id_loss = tf.math.reduce_mean(tf.math.abs(y_id - y))

        loss = gan_loss + lambda1 * fw_loss + lambda2 * bw_loss + ID_LAMBDA * id_loss

        if verbose:
            return loss, gan_loss, fw_loss, bw_loss, id_loss
        
        return loss



# === Discriminator ===

class DImageGAN(tf.Module):
    '''
    Implementation of the ImageGAN discriminator

    Reference:
        https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L271-L300
    '''
    def __init__(self, name=None):
        super(DImageGAN, self).__init__(name=auto_naming(self, name))

        # layer definitions
        # in default, ngf = 64
        # input shape: [batch, 256, 256, 3]
        self.downsample_spec = [dict(n_kernel=64, size=4, apply_norm=False),     # output shape: [batch, 128, 128, nfg]
                                dict(n_kernel=128, size=4, norm_type=BatchNorm), # output shape: [batch, 64, 64, nfg*2]
                                dict(n_kernel=256, size=4, norm_type=BatchNorm), # output shape: [batch, 32, 32, nfg*4]
                                dict(n_kernel=512, size=4, norm_type=BatchNorm), # output shape: [batch, 16, 16, nfg*8]
                                dict(n_kernel=512, size=4, norm_type=BatchNorm), # output shape: [batch, 8, 8, nfg*8]
                                dict(n_kernel=512, size=4, norm_type=BatchNorm), # output shape: [batch, 4, 4, nfg*8]
                                dict(n_kernel=512, size=4, norm_type=BatchNorm), # output shape: [batch, 2, 2, nfg*8] (mission from the original implementation)
                                dict(n_kernel=1, size=4, apply_norm=False, apply_activ=False)] # output shape: [batch, 1, 1, 1]

        # create layers
        with self.name_scope:
            self.downsamples = [DownSample(**spec) for spec in self.downsample_spec]


    def __call__(self, input, training=True):

        #outputs = []

        output = input
        # input shape: 256 x 256 x 3
        for layer in self.downsamples:
            output = layer(output, training=training)
            # DEBUG:
            #outputs.append(output)

        # DEBUG: check network outputs
        #LOG.debug('training = {}, scope = {}'.format(training, self.downsamples[-1].name_scope.name))
        #LOG.debug('input = \n{}'.format(outputs[-2].numpy()[0, :, :, :25])) #2*2*25
        #LOG.debug('output = \n{}'.format(outputs[-1].numpy()[0, :, :, :])) # 1*1*1

        # output shape: 1 x 1 x 1
        return output


class DBasic(tf.Module):
    '''
    Implmementation of the defineD_basic discriminator
    
    Reference:
        https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L338-L384
    '''
    def __init__(self, name=None):
        super(DBasic, self).__init__(name=auto_naming(self, name))

        # create sub-modules under the name scope
        with self.name_scope:

            # create layers
            # in default, nfg = 64
            # input shape: [batch, 256, 256, 3]
            self.down1 = DownSample(64, 4, apply_norm=False) # output shape: [batch, 128, 128, nfg]
            self.down2 = DownSample(128, 4) # output shape: [batch, 64, 64, nfg*2]
            self.down3 = DownSample(256, 4) # output shape: [batch, 32, 32, nfg*4]
            
            self.pad1 = Padding()           # output shape: [batch, 34, 34, nfg*4]
            self.down4 = DownSample(512, 4, stride=1) # output shape: [batch, 34, 34, nfg*8]

            self.pad2 = Padding()           # output shape: [batch, 36, 36, nfg * 8]
            self.conv2 = Conv(1, 4, stride=1, is_biased=True) # output shape: [batch, 36, 36, 1]




    def __call__(self, input, training=True):

        output = self.down1(input, training=training)
        output = self.down2(output, training=training)
        output = self.down3(output, training=training)

        output = self.pad1(output, training=training)
        # DEBUG:
        #out_temp = self.down4(output, training=training)
        output = self.down4(output, training=training)

        # DEBUG:
        #output = self.pad2(out_temp, training=training)
        #output = self.conv2(output, training=training)
        output = self.pad2(output, training=training)
        output = self.conv2(output, training=training)

        # DEBUG: check network outputs
        #LOG.debug('training = {}, scope = {}'.format(training, self.conv2.name_scope.name))
        #LOG.debug('input = \n{}'.format(out_temp.numpy()[0, :1, :1, :100])) #1*36*100
        #LOG.debug('output = \n{}'.format(output.numpy()[0, :1, :, 0])) # 1*36*1

        return output


class Discriminator(tf.Module):
    def __init__(self, net=DBasic, loss_type='BCE', name=None):
        '''
        Discriminator network

        Args:
            net: network type
            loss_type: either ['BCE', 'MSE', 'ABS']
        '''
        super(Discriminator, self).__init__(name=auto_naming(self, name))

        if loss_type == 'BCE':
            self._loss_func = self._binary_cross_entropy_loss
        elif loss_type == 'MSE':
            self._loss_func = self._mean_square_error_loss
        elif loss_type == 'ABS':
            self._loss_func = self._abs_loss
        else:
            raise NotImplementedError('The loss function for the loss type \'{}\' is not implemented'.format(loss_type))

        with self.name_scope:
            self.net = net()

    @tf.function
    @tf.Module.with_name_scope
    def __call__(self, input, training=True):
        
        return self.net(input, training=training)


    def _binary_cross_entropy_loss(self, y_pred, x_pred):
        '''
        Discriminator binary cross entropy loss

        Args:
            y_pred: D(y)
            x_pred: D(G(x))

        Reference:
            https://developers.google.com/machine-learning/gan/loss#minimax-loss
            https://github.com/junyanz/CycleGAN/blob/master/models/cycle_gan_model.lua#L136-L153
        '''

        real_loss = tf.math.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.ones_like(y_pred), logits=y_pred))

        gened_loss = tf.math.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.zeros_like(x_pred), logits=x_pred))

        loss = (real_loss + gened_loss) * 0.5
        return loss

    def _mean_square_error_loss(self, y_pred, x_pred):
        '''
        Discriminator mean square error loss

        Args:
            y_pred: D(y)
            x_pred: D(G(x))

        Reference:
            https://github.com/junyanz/CycleGAN/blob/master/models/cycle_gan_model.lua#L136-L153
        '''

        real_loss = tf.math.reduce_mean(
                        tf.math.squared_difference(
                            tf.math.sigmoid(y_pred), tf.ones_like(y_pred)) )

        # we dont use squared_difference, since the target is ZERO: 
        #     loss = mean(pow(x) - pow(0))
        gened_loss = tf.math.reduce_mean(tf.math.square(tf.math.sigmoid(x_pred)))

        loss = (real_loss + gened_loss) * 0.5
        return loss

    def loss(self, *args, **kwargs):

        return self._loss_func(*args, **kwargs)


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
        # [-1.0 ~ 1.0]
        img = tf.image.convert_image_dtype(img, tf.float32) * 2.0 - 1.0
        return img

    def process_image(is_train):
        if is_train:
            def _process_data(img):

                img = tf.image.resize(img, [int(IMAGE_HEIGHT*1.12), int(IMAGE_WIDTH*1.12)])

                # random crop
                img = tf.image.random_crop(img, size=[IMAGE_HEIGHT, IMAGE_WIDTH, 3])

                # random flip
                if np.random.random() > 0.5:
                    img = tf.image.flip_left_right(img)

                return tf.reshape(img, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        else:
            def _process_data(img):
                
                img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
                return tf.reshape(img, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

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


def inference(input, gen, preprocess=True):
    '''
    Args:
        input: input image (IMAGE_HEIGHT, IMAGE_WIDTH, 3), the dimension must be wither 3D or 4D
    '''

    # get shape
    ndim = len(input.shape)
    if ndim == 3:
        input_shape = input.shape[0:2]
    elif ndim == 4:
        input_shape = input.shape[1:3]
    else:
        raise ValueError('Unknown shape of image with dimension: {}, only accept 3D or 4D images'.format(ndim))

    input_dtype = input.dtype

    if preprocess:
        input = tf.image.convert_image_dtype(input, tf.float32) * 2.0 - 1.0

    # resize image
    x = tf.image.resize(input, [IMAGE_HEIGHT, IMAGE_WIDTH])
    x_shape = x.shape

    # reshape to 4D image
    x = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    # generate
    y_gen = gen(x, training=False)
    # reshape to the original shape
    y_gen= tf.reshape(y_gen, x_shape)

    # resize to the original size
    y_gen = tf.image.resize(y_gen, input_shape)

    if preprocess:
        # convert to original dtype
        y_gen = tf.image.convert_image_dtype(y_gen*0.5+0.5, input_dtype)

    return y_gen.numpy()

@tf.function
def test_step(a, b, genAB, genBA, disAB, disBA):
    # generate
    b_gen = genAB(a, training=False) #G(x)
    a_gen = genBA(b, training=False) #
    a_regen = genBA(b_gen, training=False)
    b_regen = genAB(a_gen, training=False)
    a_id = genBA(a, training=False)
    b_id = genAB(b, training=False)

    # discriminate
    b_pred = disAB(b, training=False)
    a_pred = disBA(a, training=False)
    gen_b_pred = disAB(b_gen, training=False)
    gen_a_pred = disBA(a_gen, training=False)
    
    #----------------------------------    x, y, F(G(x)), G(F(y)), D(G(x)),    G(y), lambda1,        lambda2)
    gen_ab_loss, *gen_ab_misc = genAB.loss(a, b, a_regen, b_regen, gen_b_pred, b_id, CYCLE_A_LAMBDA, CYCLE_B_LAMBDA, verbose=VERBOSE)
    gen_ba_loss, *gen_ba_misc = genBA.loss(b, a, b_regen, a_regen, gen_a_pred, a_id, CYCLE_B_LAMBDA, CYCLE_A_LAMBDA, verbose=VERBOSE)
    #----------------------  D(y),   D(G(x))
    dis_ab_loss = disAB.loss(b_pred, gen_b_pred)
    dis_ba_loss = disBA.loss(a_pred, gen_a_pred)

    return gen_ab_loss, gen_ba_loss, dis_ab_loss, dis_ba_loss, gen_ab_misc, gen_ba_misc

def test(genAB, genBA, disAB, disBA, test_setA, test_setB):
    
    LOG.info('Evaluating model...')
    start = time.time()

    total_gen_ab_loss = []
    total_gen_ba_loss = []
    total_dis_ab_loss = []
    total_dis_ba_loss = []
    total_aba_cycle_consis_loss = []
    total_bab_cycle_consis_loss = []
    total_a_identity_loss = []
    total_b_identity_loss = []

    for step, (dataA, dataB) in enumerate(zip(test_setA, test_setB)):
        # forward 
        (gen_ab_loss, gen_ba_loss, dis_ab_loss, dis_ba_loss, *misc) = test_step(dataA, dataB, genAB, genBA, disAB, disBA)

        total_gen_ab_loss.append(gen_ab_loss.numpy())
        total_gen_ba_loss.append(gen_ba_loss.numpy())
        total_dis_ab_loss.append(dis_ab_loss.numpy())
        total_dis_ba_loss.append(dis_ba_loss.numpy())

        if VERBOSE:

            total_aba_cycle_consis_loss.append(misc[0][1].numpy())
            total_bab_cycle_consis_loss.append(misc[0][2].numpy())
            total_a_identity_loss.append(misc[0][3].numpy())
            total_b_identity_loss.append(misc[1][3].numpy())

    end = time.time()
    elapsed_time = datetime.timedelta(seconds=end-start)

    avg_gen_ab_loss = np.array(total_gen_ab_loss).mean()
    avg_gen_ba_loss = np.array(total_gen_ba_loss).mean()
    avg_dis_ab_loss = np.array(total_dis_ab_loss).mean()
    avg_dis_ba_loss = np.array(total_dis_ba_loss).mean()

    if VERBOSE:
        avg_aba_cycle_consis_loss = np.array(total_aba_cycle_consis_loss).mean()
        avg_bab_cycle_consis_loss = np.array(total_bab_cycle_consis_loss).mean()
        avg_a_identity_loss = np.array(total_a_identity_loss).mean()
        avg_b_identity_loss = np.array(total_b_identity_loss).mean()
    
    LOG.set_header('Test')
    LOG.add_row('Time elapsed', elapsed_time, fmt='{}: {}')
    if VERBOSE:
        LOG.add_row('Avg ABA-cycle consis loss', avg_aba_cycle_consis_loss, fmt='{}: {:.6f}')
        LOG.add_row('Avg BAB-cycle consis loss', avg_bab_cycle_consis_loss, fmt='{}: {:.6f}')
        LOG.add_row('Avg A identity loss', avg_a_identity_loss, fmt='{}: {:.6f}')
        LOG.add_row('Avg B identity loss', avg_b_identity_loss, fmt='{}: {:.6f}')
    LOG.add_row('Avg Gen AB loss', avg_gen_ab_loss, fmt='{}: {:.6f}')
    LOG.add_row('Avg Gen BA loss', avg_gen_ba_loss, fmt='{}: {:.6f}')
    LOG.add_row('Avg Dis AB loss', avg_dis_ab_loss, fmt='{}: {:.6f}')
    LOG.add_row('Avg Dis BA loss', avg_dis_ba_loss, fmt='{}: {:.6f}')
    LOG.flush('INFO')

@tf.function
def train_step(a, b, genAB, genBA, disAB, disBA, genAB_opt, genBA_opt, disAB_opt, disBA_opt):

    # since I calling gradient() for twice, `persistent` must be set to True
    with tf.GradientTape(persistent=True) as tape:

        # generate
        b_gen = genAB(a, training=True) # G(x)
        a_gen = genBA(b, training=True) # F(y)
        a_regen = genBA(b_gen, training=True) # F(G(x))
        b_regen = genAB(a_gen, training=True) # G(F(y))
        a_id = genBA(a, training=True) # F(x)
        b_id = genAB(b, training=True) # G(y)

        # discriminate
        b_pred = disAB(b, training=True) # D(y)=True
        a_pred = disBA(a, training=True) # H(x)=True
        gen_b_pred = disAB(b_gen, training=True) # D(G(x))=False
        gen_a_pred = disBA(a_gen, training=True) # H(F(y))=False
        
        #------------------------------------  x, y, F(G(x)), G(F(y)), D(G(x)),    G(y), lambda1,        lambda2)
        gen_ab_loss, *gen_ab_misc = genAB.loss(a, b, a_regen, b_regen, gen_b_pred, b_id, CYCLE_A_LAMBDA, CYCLE_B_LAMBDA, verbose=VERBOSE)
        gen_ba_loss, *gen_ba_misc = genBA.loss(b, a, b_regen, a_regen, gen_a_pred, a_id, CYCLE_B_LAMBDA, CYCLE_A_LAMBDA, verbose=VERBOSE)
        #----------------------  D(y),   D(G(x))
        dis_ab_loss = disAB.loss(b_pred, gen_b_pred)
        dis_ba_loss = disBA.loss(a_pred, gen_a_pred)

    # compute gradients
    gen_ab_grad = tape.gradient(gen_ab_loss, genAB.trainable_variables)
    gen_ba_grad = tape.gradient(gen_ba_loss, genBA.trainable_variables)
    dis_ab_grad = tape.gradient(dis_ab_loss, disAB.trainable_variables)
    dis_ba_grad = tape.gradient(dis_ba_loss, disBA.trainable_variables)

    # apply gradients
    genAB_opt.apply_gradients(zip(gen_ab_grad, genAB.trainable_variables))
    genBA_opt.apply_gradients(zip(gen_ba_grad, genBA.trainable_variables))
    disAB_opt.apply_gradients(zip(dis_ab_grad, disAB.trainable_variables))
    disBA_opt.apply_gradients(zip(dis_ba_grad, disBA.trainable_variables))

    return gen_ab_loss, gen_ba_loss, dis_ab_loss, dis_ba_loss, gen_ab_misc, gen_ba_misc


def train(genAB, genBA, disAB, disBA, genAB_opt, genBA_opt, disAB_opt, disBA_opt, 
            manager, train_setA, train_setB, test_setA, test_setB):

    LOG.info('Start training for {} epochs'.format(EPOCHS))

    def plot_sample(a, b, fname_suffix=''):

        def plot(a, b, a_gen, b_gen, fname):

            plt.figure(figsize=(10, 10))

            def add_subplot(img, title, axis):
                plt.subplot(2, 2, axis, frameon=False)
                plt.title(title, fontsize=24)
                plt.imshow(img * 0.5+0.5)
                plt.axis('off')

            add_subplot(a, LABELS[0], 1)
            add_subplot(b, LABELS[1], 2)
            add_subplot(b_gen, LABELS[2], 3)
            add_subplot(a_gen, LABELS[3], 4)

            plt.tight_layout()

            os.makedirs(os.path.dirname(fname), exist_ok=True)
            plt.savefig(fname)
            # close figure
            plt.close()
            LOG.info('[Image Saved] save to: {}'.format(fname))
        
        # if data is a tensor, convert it to a numpy array
        if tf.is_tensor(a) and hasattr(a, 'numpy'):
            a = a.numpy()

        if tf.is_tensor(b) and hasattr(b, 'numpy'):
            b = b.numpy()

        # reshape images
        a = a.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        b = b.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        # generate images
        b_gen = inference(a, genAB, preprocess=False)
        a_gen = inference(b, genBA, preprocess=False)

        plot_path = os.path.join(MODEL_DIR, 'images/{}_epoch_{}.png'.format(MODEL_NAME, fname_suffix))
        plot(a, b, a_gen, b_gen, plot_path)


    # take one sample from training set A to draw and evaluate results
    for one_trainA in train_setA.take(1): pass
    one_trainA = one_trainA[0]

    # take one sample from training set B to draw and evaluate results
    for one_trainB in train_setB.take(1): pass
    one_trainB = one_trainB[0]

    # take one sample from test set A to draw and evaluate results
    for one_testA in test_setA.take(1): pass
    one_testA = one_testA[0]

    # take one sample from test set B to draw and evaluate results
    for one_testB in test_setB.take(1): pass
    one_testB = one_testB[0]

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        total_gen_ab_loss = []
        total_gen_ba_loss = []
        total_dis_ab_loss = []
        total_dis_ba_loss = []
        total_aba_cycle_consis_loss = []
        total_bab_cycle_consis_loss = []
        total_a_identity_loss = []
        total_b_identity_loss = []

        step_start = epoch_start
        for step, (dataA, dataB) in enumerate(zip(train_setA, train_setB)):

            (gen_ab_loss, gen_ba_loss, dis_ab_loss, dis_ba_loss, *misc) = train_step(dataA, dataB, genAB, genBA, disAB, disBA, 
                                                                                   genAB_opt, genBA_opt, disAB_opt, disBA_opt)
            
            total_gen_ab_loss.append(gen_ab_loss.numpy())
            total_gen_ba_loss.append(gen_ba_loss.numpy())
            total_dis_ab_loss.append(dis_ab_loss.numpy())
            total_dis_ba_loss.append(dis_ba_loss.numpy())

            if VERBOSE:

                total_aba_cycle_consis_loss.append(misc[0][1].numpy())
                total_bab_cycle_consis_loss.append(misc[0][2].numpy())
                total_a_identity_loss.append(misc[0][3].numpy())
                total_b_identity_loss.append(misc[1][3].numpy())

                if (step+1) % 100 == 0:
                    step_end = time.time()
                    step_elapsed_time = datetime.timedelta(seconds=step_end-step_start)
                    # === print LOG ===
                    LOG.set_header('Epoch {}/{}'.format(epoch+1, EPOCHS))
                    LOG.add_row('Step', step+1)
                    LOG.add_row('ABA-cycle consis loss', misc[0][1], fmt='{}: {:.6f}')
                    LOG.add_row('BAB-cycle consis loss', misc[0][2], fmt='{}: {:.6f}')
                    LOG.add_row('A identity loss', misc[0][3], fmt='{}: {:.6f}')
                    LOG.add_row('B identity loss', misc[1][3], fmt='{}: {:.6f}')
                    LOG.add_row('Gen AB loss', gen_ab_loss, fmt='{}: {:.6f}')
                    LOG.add_row('Gen BA loss', gen_ba_loss, fmt='{}: {:.6f}')
                    LOG.add_row('Dis AB loss', dis_ab_loss, fmt='{}: {:.6f}')
                    LOG.add_row('Dis BA loss', dis_ba_loss, fmt='{}: {:.6f}')
                    LOG.flush('INFO')
                    # =================

        epoch_end = time.time()

        elapsed_time = datetime.timedelta(seconds=epoch_end-epoch_start)
        avg_gen_ab_loss = np.array(total_gen_ab_loss).mean()
        avg_gen_ba_loss = np.array(total_gen_ba_loss).mean()
        avg_dis_ab_loss = np.array(total_dis_ab_loss).mean()
        avg_dis_ba_loss = np.array(total_dis_ba_loss).mean()

        if VERBOSE:
            avg_aba_cycle_consis_loss = np.array(total_aba_cycle_consis_loss).mean()
            avg_bab_cycle_consis_loss = np.array(total_bab_cycle_consis_loss).mean()
            avg_a_identity_loss = np.array(total_a_identity_loss).mean()
            avg_b_identity_loss = np.array(total_b_identity_loss).mean()

        # === print LOG === 
        LOG.set_header('Epoch {}/{}'.format(epoch+1, EPOCHS))
        LOG.add_row('Time elapsed', elapsed_time, fmt='{}: {}')
        if VERBOSE:
            LOG.add_row('Avg ABA-cycle consis loss', avg_aba_cycle_consis_loss, fmt='{}: {:.6f}')
            LOG.add_row('Avg BAB-cycle consis loss', avg_bab_cycle_consis_loss, fmt='{}: {:.6f}')
            LOG.add_row('Avg A identity loss', avg_a_identity_loss, fmt='{}: {:.6f}')
            LOG.add_row('Avg B identity loss', avg_b_identity_loss, fmt='{}: {:.6f}')
        LOG.add_row('Avg Gen AB loss', avg_gen_ab_loss, fmt='{}: {:.6f}')
        LOG.add_row('Avg Gen BA loss', avg_gen_ba_loss, fmt='{}: {:.6f}')
        LOG.add_row('Avg Dis AB loss', avg_dis_ab_loss, fmt='{}: {:.6f}')
        LOG.add_row('Avg Dis BA loss', avg_dis_ba_loss, fmt='{}: {:.6f}')
        LOG.flush('INFO')
        # ==================

        if (EVAL_EPOCHS > 0 and 
                (epoch+1) % EVAL_EPOCHS == 0):
            
            test(genAB, genBA, disAB, disBA, test_setA, test_setB)
            plot_sample(one_trainA, one_trainB, '{}_train'.format(epoch+1))
            plot_sample(one_testA, one_testB, '{}_test'.format(epoch+1))

        if (SAVE_EPOCHS > 0 and
                (epoch+1) % SAVE_EPOCHS == 0):
            
            save_path = manager.save()
            LOG.info('[Model Saved] save to: {}'.format(save_path))

    test(genAB, genBA, disAB, disBA, test_setA, test_setB)
    plot_sample(one_trainA, one_trainB, 'final_train')
    plot_sample(one_testA, one_testB, 'final_test')

def initialize_modules(genAB, genBA, disAB, disBA, training=False):
    genAB(np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32), training=training)
    genBA(np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32), training=training)
    disAB(np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32), training=training)
    disBA(np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32), training=training)


if __name__ == '__main__':

    # download dataset
    data_filename = maybe_download_dataset(DATA_FILENAME,
                                           origin=DATA_URL,
                                           extract=True)
    data_path = os.path.dirname(data_filename)

    # create generators and discriminators
    genAB = Generator(net=UNet256, name='genAB') # A -> B
    genBA = Generator(net=UNet256, name='genBA') # B -> A
    disAB = Discriminator(net=DBasic, name='disAB')
    disBA = Discriminator(net=DBasic, name='disBA')

    # create optimizers
    genAB_opt = tf.optimizers.Adam(learning_rate=GEN_LR, beta_1=0.5)
    genBA_opt = tf.optimizers.Adam(learning_rate=GEN_LR, beta_1=0.5)
    disAB_opt = tf.optimizers.Adam(learning_rate=DIS_LR, beta_1=0.5)
    disBA_opt = tf.optimizers.Adam(learning_rate=DIS_LR, beta_1=0.5)

    # checkpoint
    checkpoint= tf.train.Checkpoint(generatorAB=genAB,
                                    generatorBA=genBA,
                                    discriminatorAB=disAB,
                                    discriminatorBA=disBA,
                                    genAB_optimizer=genAB_opt,
                                    genBA_optimizer=genBA_opt,
                                    disAB_optimizer=disAB_opt,
                                    disBA_optimizer=disBA_opt)

    # checkpoint manager
    manager = tf.train.CheckpointManager(checkpoint, MODEL_DIR, max_to_keep=3, checkpoint_name=MODEL_NAME)

    # initialize modules
    initialize_modules(genAB, genBA, disAB, disBA, training=TRAIN)


    if manager.latest_checkpoint is not None:
        LOG.debug('Restore checkpoint from: {}'.format(manager.latest_checkpoint))
    
    # restore checkpoint
    status = checkpoint.restore(manager.latest_checkpoint)

    if not INFERENCE:
        test_setA = create_dataset(path=os.path.join(data_path, os.path.join(TEST_A_PATH, TEST_A_FILE)),
                                    is_train=False)
        test_setB = create_dataset(path=os.path.join(data_path, os.path.join(TEST_B_PATH, TEST_B_FILE)),
                                    is_train=False)

    if TRAIN:
        # training mode
        
        train_setA = create_dataset(path=os.path.join(data_path, os.path.join(TRAIN_A_PATH, TRAIN_A_FILE)),
                                    is_train=True)
        train_setB = create_dataset(path=os.path.join(data_path, os.path.join(TRAIN_B_PATH, TRAIN_B_FILE)),
                                    is_train=True)

        train(genAB, genBA, disAB, disBA, 
              genAB_opt, genBA_opt, disAB_opt, disBA_opt, 
              manager, 
              train_setA, train_setB,
              test_setA, test_setB)

    elif INFERENCE:
        # inference mode
        pass
    
    else:
        # test mode
        test(genAB, genBA, disAB, disBA, 
            test_setA, test_setB)