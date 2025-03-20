import sys
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from common_methods import read_name_list
import numpy as np
np.random.seed(1)
tf.set_random_seed(1)
from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = device_type_dict[device_type]["device_id"]

import read_data_plm as rd
import save_result as sh



class MLP(object):

    def __init__(self, workdir, go_type, round_index):

        self.go_type = go_type

        self.test_feature, self.test_label, self.test_name_list = rd.read_data(workdir, go_type)
        self.model_dir = os.path.join(plm_data_dir, go_type, plm_model_name, str(round_index))
        self.result_dir = os.path.join(workdir, go_type, plm_result_dir, plm_round_name + str(round_index))
        self.term_list_file = os.path.join(plm_data_dir, go_type, plm_term_list)
        self.term_list = read_name_list(self.term_list_file)



    def dnn(self, x, keep_prob, is_train):   # deep neural network

        x = tf.layers.dense(inputs=x, units=plm_full_connect_number, activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.batch_normalization(x, training=is_train)
        y_pred = tf.layers.dense(inputs=x, units=go_label_size[self.go_type], activation=tf.nn.sigmoid)

        return y_pred

    def process(self, x, y, keep_prob, is_train):   # process

        with tf.name_scope('embedding'):

            y_pred = self.dnn(x, keep_prob, is_train)

        with tf.name_scope('caculate_loss'):

            cross_entropy = y * tf.log(y_pred + 1e-6) + (1 - y) * tf.log(1 + 1e-6 - y_pred)
            cross_entropy = -tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(plm_learning_rate).minimize(cross_entropy)

        return train_step, cross_entropy, y_pred


    def running(self): # main process

        tf.reset_default_graph()
        tf.global_variables_initializer()

        x = tf.placeholder(tf.float32, [None, plm_feature_size])
        y = tf.placeholder(tf.float32, [None, go_label_size[self.go_type]])
        keep_prob = tf.placeholder(tf.float32)
        is_train = tf.placeholder(tf.bool)

        train_step, cross_entropy, y_pred = self.process(x, y, keep_prob, is_train)

        test_data_list = rd.create_batch(self.test_feature, self.test_label, self.test_name_list, plm_batch_size, False)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = device_type_dict["gpu"]["gpu_ratio"]

        with tf.Session(config=config) as sess:

            ckpt = tf.train.latest_checkpoint(self.model_dir)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)
            print("successful load")

            test_pred_matrix = np.zeros([1, go_label_size[self.go_type]])

            for test_data in test_data_list:

                sub_test_feature, sub_test_label, sub_test_name = test_data
                current_y_pred = sess.run(y_pred, feed_dict={x: sub_test_feature, y: sub_test_label, keep_prob: 1, is_train: False})
                test_pred_matrix = np.concatenate((test_pred_matrix, current_y_pred), axis=0)

            test_pred_matrix = test_pred_matrix[1:]

            sh.save_cross_entropy_results(self.result_dir, self.go_type, self.term_list_file, plm_result_name, self.test_name_list, test_pred_matrix)




















