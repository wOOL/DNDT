import tensorflow as tf
from functools import reduce


def tf_kron_prod(a, b):
    res = tf.einsum('ij,ik->ijk', a, b)
    res = tf.reshape(res, [-1, tf.reduce_prod(res.shape[1:])])
    return res


def tf_bin(x, cut_points, temperature=0.1):
    # x is a N-by-1 matrix (column vector)
    # cut_points is a D-dim vector (D is the number of cut-points)
    # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
    D = cut_points.get_shape().as_list()[0]
    W = tf.reshape(tf.linspace(1.0, D + 1.0, D + 1), [1, -1])
    cut_points = tf.contrib.framework.sort(cut_points)  # make sure cut_points is monotonically increasing
    b = tf.cumsum(tf.concat([tf.constant(0.0, shape=[1]), -cut_points], 0))
    h = tf.matmul(x, W) + b
    res = tf.nn.softmax(h / temperature)
    return res


def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1):
    # cut_points_list contains the cut_points for each dimension of feature
    leaf = reduce(tf_kron_prod,
                  map(lambda z: tf_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list)))
    return tf.matmul(leaf, leaf_score)
