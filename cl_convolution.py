import math

import cv2
import numpy as np
from tensorflow.python import pywrap_tensorflow

def get_variables(image, variables, share_weights=False, stride=1, padding="SAME"):
    kernel, gamma, beta = variables
    # kernel = variables[0]
    conv_times = kernel.shape[3]
    if share_weights == False:
        assert image.shape[3] == kernel.shape[2]
        output_height, output_width = calculate_output_mat(image, kernel.shape, stride, padding)
        output_height = int(output_height)
        output_width = int(output_width)
        if padding == "SAME":
            image_after_padding = add_padding(image, output_height, output_width, stride, kernel.shape)
        else:
            image_after_padding = crop_image(image, output_height, output_width, stride)
        out_mat = np.zeros((image.shape[0], output_height, output_width, conv_times))

        for j in range(image.shape[0]):
            for i in range(conv_times):
                out_mat[j, :, :, i] = do_conv(image_after_padding[j, :], output_height, output_width, kernel[:, :, :, i], stride)
        # test = out_mat.reshape([6, 6])
        conv_out = out_mat + 0.1
        y = batchnorm_forward(conv_out, gamma, beta, {"mode": "test", "moving_variance": 1, "moving_mean": 0})
        y_ = relu(y)
        return conv_out


def relu(x):
    # a = x.shape
    n, k = x.shape[0], x.shape[-1]
    for i in range(n):
        for j in range(k):
            x[i, :, :, j] = np.maximum(x[i, :, :, j], 0)


def do_conv(new_image, h_times, w_times, single_kernal, stride):
    k_size = single_kernal.shape[0]
    out_mat = np.zeros((h_times, w_times))
    for i in range(h_times):
        y = int(i * stride)
        for j in range(w_times):
            x = int(j * stride)
            for k in range(single_kernal.shape[2]):
                # a= np.multiply(image[y:y + k_size, x:x + k_size, k], single_kernal[:, :, k])
                # b= np.sum(a)
                out_mat[i, j] += np.sum(np.multiply(new_image[y:y + k_size, x:x + k_size, k], single_kernal[:, :, k]))
    return out_mat


def calculate_output_mat(image, kernal_shape, stride, padding):
    if padding == "SAME":
        output_height = math.ceil(float(image.shape[1]) / float(stride))
        output_width = math.ceil(float(image.shape[2]) / float(stride))
        return output_height, output_width
    else:
        output_height = math.ceil((float(image.shape[1]) - float(kernal_shape[1]) + 1)/float(stride))
        output_width = math.ceil((float(image.shape[2]) - float(kernal_shape[2]) + 1) / float(stride))
        return int(output_height), int(output_width)


def add_padding(need_padding, height, width, stride, k):# all add to right and bottom
    h_add = (height - 1) * stride + k[0] - need_padding.shape[1]
    w_add = (width - 1) * stride + k[1] - need_padding.shape[2]
    new_h = need_padding.shape[1] + h_add
    new_w = need_padding.shape[2] + w_add
    return_mat = np.zeros((int(need_padding.shape[0]), int(new_h), int(new_w), k[2]))
    assert h_add == w_add
    for i in range(need_padding.shape[0]):
        new_mat = np.zeros((int(new_h), int(new_w), k[2]))
        new_mat[w_add-1:need_padding.shape[1] + 1, w_add-1:need_padding.shape[2] + 1] = need_padding[i, :]
        return_mat[i, :] = new_mat
    return return_mat
    # for i in range(need_padding.shape[0],need_padding.shape[0] + number_add):

    # w_add = (width -1 ) * stride + k[1] - need_padding.shape[1]


def add_bias():
    pass


def crop_image(need_crop, height, width, stride):
    raise EOFError


def batchnorm_forward_0(x, gamma, beta, eps):
    a = x.shape
    gamma = 1
    beta = 0
    # N, D = x.shape
    N = a[0]
    # 为了后向传播求导方便，这里都是分步进行的
    # step1: 计算均值
    mu = 1. / N * np.sum(x, axis=0)
    # step2: 减均值
    xmu = x - mu
    # step3: 计算方差
    sq = xmu ** 2
    var = 1. / N * np.sum(sq, axis=0)
    # step4: 计算x^的分母项
    sqrtvar = np.sqrt(var + eps)
    ivar = 1. / sqrtvar

    # step5: normalization->x^
    xhat = xmu * ivar

    # step6: scale and shift
    gammax = gamma * xhat
    out = gammax + beta

    # 存储中间变量
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

    return out, cache


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  输入:
  - x: 输入数据 shape (N, D)
  - gamma: 缩放参数 shape (D,)
  - beta: 平移参数 shape (D,)
  - bn_param: 包含如下参数的dict:
    - mode: 'train' or 'test'; 用来区分训练还是测试
    - eps: 除以方差时为了防止方差太小而导致数值计算不稳定
    - momentum: 前面讨论的momentum.
    - running_mean: 数组 shape (D,) 记录最新的均值
    - running_var 数组 shape (D,) 记录最新的方差

  返回一个tuple:
  - out: shape (N, D)
  - cache: 缓存反向计算时需要的变量
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-3)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape[0], x.shape[1:]
  running_mean = bn_param.get('moving_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('moving_variance', np.ones(D, dtype=x.dtype))

  # out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    x_mean = x.mean(axis=0)
    x_var = x.var(axis=0)
    x_normalized = (x-x_mean)/np.sqrt(x_var+eps)
    out = gamma * x_normalized + beta

    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var
    cache = (x, x_mean, x_var, x_normalized, beta, gamma, eps)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x_normalized = (x - running_mean)/np.sqrt(running_var +eps)
    out = gamma*x_normalized + beta

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out
import time
begin = time.time()
test_mat = np.zeros((2, 112, 112, 3))
test_mat[0, :] = cv2.resize(cv2.imread('15.bmp'), (112, 112))
test_mat[1, :] = cv2.resize(cv2.imread('4.bmp'), (112, 112))

reader = pywrap_tensorflow.NewCheckpointReader("test_ckpt/test.ckpt")
var_to_shape_map = reader.get_variable_to_shape_map()
for key in sorted(var_to_shape_map):
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
kernel = reader.get_tensor('Variable')
gamma = reader.get_tensor('batch_normalization/gamma')
beta = reader.get_tensor('batch_normalization/beta')
variable = [kernel, gamma, beta]

# variable = [kernel]

# kernal.tofile("test.bin")
# kernal = np.fromfile("test.bin", dtype=np.float32)
# kernal.shape = 3, 3, 3, 5
# print((a == kernal).all())
# test_mat = np.arange(1, 37, 1, float)
# kernal= np.arange(1, 10, 1, float)
# test_mat = test_mat.reshape([1, 6, 6, 1])
# kernal = kernal.reshape([3, 3, 1, 1])
output = get_variables(test_mat, variable)
outputs = np.squeeze(output)
# print(time.time() - begin)
import test
y = np.squeeze(test.restore())
# print((outputs == y).all())
print(time.time() - begin)
print('end')
# h, w = calculate_output_mat(test_img, (3, 3), 2, "SAME")
# add_padding(test_img, h, w, 2, (3, 3))