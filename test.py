import unittest
import numpy as np
import tensorflow as tf
from neural_network_decision_tree import tf_kron_prod, tf_bin, nn_decision_tree


class TestNeuralNetworkDecisionTree(unittest.TestCase):
    def test_tf_kron_prod(self):
        a = np.array([[1, 2],
                      [3, 4]]).astype(np.float32)
        b = np.array([[0.1, 0.2],
                      [0.3, 0.4]]).astype(np.float32)
        sess = tf.InteractiveSession()
        res = tf_kron_prod(tf.constant(a), tf.constant(b)).eval()
        exp = np.array([[0.1, 0.2, 0.2, 0.4],
                        [0.9, 1.2, 1.2, 1.6, ]])
        np.testing.assert_almost_equal(res, exp)

    def test_tf_bin(self):
        x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1).astype(np.float32)
        cut = np.array([1.5, 3.5]).astype(np.float32)
        sess = tf.InteractiveSession()
        res = tf_bin(tf.constant(x), tf.constant(cut)).eval()
        exp = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 1]])
        np.testing.assert_almost_equal(res, exp, decimal=2)

    def test_nn_decision_tree(self):
        x = np.array([[1, 2],
                      [2, 3],
                      [1, 3],
                      [2, 2]]).astype(np.float32)
        cut_points_list = [np.array([1.5]).astype(np.float32),
                           np.array([2.5]).astype(np.float32)]
        leaf_score = np.array([[4, 1],
                               [3, 2],
                               [2, 3],
                               [1, 4]]).astype(np.float32)
        sess = tf.InteractiveSession()
        res = nn_decision_tree(tf.constant(x),
                               [tf.constant(i) for i in cut_points_list],
                               tf.constant(leaf_score)).eval()
        exp = np.array([[4, 1],
                        [1, 4],
                        [3, 2],
                        [2, 3]])
        np.testing.assert_almost_equal(res, exp, decimal=1)


if __name__ == '__main__':
    unittest.main()
