import sys
import os
import read_data_hand_craft as rd
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()
from config import *
import Triplet_Loss as tl
import save_result as sh
from common_methods import read_name_list
os.environ["CUDA_VISIBLE_DEVICES"] = device_type_dict[device_type]["device_id"]

class LSTM(object):

    def __init__(self, workdir, go_type, round_index):

        self.train_name_list_file = os.path.join(hc_data_dir, go_type, hc_train_name)
        self.train_label_onehot_file = os.path.join(hc_data_dir, go_type, hc_train_label_onehot)
        self.train_label_file = os.path.join(hc_data_dir, go_type, hc_train_label)
        self.term_list_file = os.path.join(hc_data_dir, go_type, hc_term_list)
        self.train_seq_length_file = os.path.join(hc_train_feature_dir, hc_train_seq_length)

        self.test_name_list_file = os.path.join(workdir, hc_workspace, go_type, hc_test_name)
        self.test_label_onehot_file = os.path.join(workdir, hc_workspace, go_type, hc_test_label_onehot)
        self.test_seq_length_file = os.path.join(workdir, hc_workspace, go_type, hc_test_seq_length)


        test_pssm_feature_dir = os.path.join(workdir, blast_workspace, blast_pssm_array_dir)
        test_ss_feature_dir = os.path.join(workdir, ss_workspace, ss_array_dir)
        test_interpro_feature_dir = os.path.join(workdir, interproscan_workspace, interproscan_featuredir)


        '''
        self.train_feature_array1, self.train_feature_array2, self.train_feature_array3,\
        self.train_label_array, self.train_name_list, self.train_sequence_length_dict = \
            rd.read_data(hc_train_pssm_feature_dir, hc_train_ss_feature_dir, hc_train_interpro_feature_dir,
                         self.train_name_list_file, self.train_label_onehot_file, self.train_seq_length_file)
        '''


        self.test_feature_array1, self.test_feature_array2, self.test_feature_array3, \
        self.test_label_array, self.test_name_list, self.test_sequence_length_dict = \
            rd.read_data(test_pssm_feature_dir, test_ss_feature_dir, test_interpro_feature_dir,
                         self.test_name_list_file, self.test_label_onehot_file, self.test_seq_length_file)

        self.model_dir = os.path.join(hc_data_dir, go_type, hc_model_name, str(round_index))
        self.go_type = go_type
        self.round_index = round_index


        self.cross_entropy_result_dir = os.path.join(workdir, hc_workspace, go_type, hc_cross_entropy_dir, hc_round_name + str(round_index))
        self.distance_dir = os.path.join(workdir, hc_workspace, go_type, hc_distance_dir, hc_round_name + str(round_index))
        self.train_embedding_file = os.path.join(hc_data_dir, go_type, hc_train_embedding_dir, hc_train_embeddings_name + str(round_index) + ".npy")

        self.train_name_list = read_name_list(os.path.join(hc_data_dir, go_type, hc_train_name_random_dir, hc_train_name_random + str(round_index)))


    def double_LSTM(self, x, seq_length):  # double direction LSTM

        cell_fw_lstm_cells = tf.nn.rnn_cell.LSTMCell(hc_rnn_unit, name = "cell1")
        cell_bw_lstm_cells = tf.nn.rnn_cell.LSTMCell(hc_rnn_unit, name = "cell2")

        initial_state_fw = cell_fw_lstm_cells.zero_state(tf.shape(x)[0], dtype=tf.float32)
        initial_state_bw = cell_bw_lstm_cells.zero_state(tf.shape(x)[0], dtype=tf.float32)

        rnn_out, states = tf.nn.bidirectional_dynamic_rnn(cell_fw_lstm_cells, cell_bw_lstm_cells, inputs=x,
                                                          initial_state_fw=initial_state_fw,
                                                          initial_state_bw=initial_state_bw, sequence_length=seq_length)

        rnn_out = tf.concat([rnn_out[0], rnn_out[1]], 2)

        return rnn_out

    def mask_attention(self, inputs, key_masks=None, type=None):
        """Masks paddings on keys or queries to inputs
        inputs: 3d tensor. (h*N, T_q, T_k)
        key_masks: 3d tensor. (N, 1, T_k)
        type: string. "key" | "future"

        e.g.,
        >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
        >> key_masks = tf.constant([[0., 0., 1.],
                                    [0., 1., 1.]])
        >> mask(inputs, key_masks=key_masks, type="key")
        array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

           [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

           [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

           [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
        """
        padding_num = -2 ** 32 + 1
        if type in ("k", "key", "keys"):
            key_masks = tf.to_float(key_masks)
            key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
            key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
            outputs = inputs + key_masks * padding_num
        # elif type in ("q", "query", "queries"):
        #     # Generate masks
        #     masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        #     masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        #     masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
        #
        #     # Apply masks to inputs
        #     outputs = inputs*masks
        elif type in ("f", "future", "right"):
            diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

            paddings = tf.ones_like(future_masks) * padding_num
            outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        else:
            print("Check if you entered type correctly!")

        return outputs

    def scaled_dot_product_attention(self, Q, K, V, key_masks, causality=False, dropout_rate=0, training=True,
                                     scope="scaled_dot_product_attention"):
        '''See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        key_masks: A 2d tensor with shape of [N, key_seqlen]
        causality: If True, applies masking for future blinding
        dropout_rate: A floating point number of [0, 1].
        training: boolean for controlling droput
        scope: Optional scope for `variable_scope`.
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

            # scale
            outputs /= d_k ** 0.5

            # key masking
            outputs = self.mask_attention(outputs, key_masks=key_masks, type="key")

            # causality or future blinding masking
            if causality:
                outputs = self.mask_attention(outputs, type="future")

            # softmax
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            # tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

            # # query masking
            # outputs = mask(outputs, Q, K, type="query")

            # dropout
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

    def multihead_attention(self, queries, keys, values, key_masks,
                            num_heads=8,
                            dropout_rate=0,
                            training=True,
                            causality=False,
                            scope="multihead_attention"):
        '''Applies multihead attention. See 3.2.2
        queries: A 3d tensor with shape of [N, T_q, d_model].
        keys: A 3d tensor with shape of [N, T_k, d_model].
        values: A 3d tensor with shape of [N, T_k, d_model].
        key_masks: A 2d tensor with shape of [N, key_seqlen]
        num_heads: An int. Number of heads.
        dropout_rate: A floating point number.
        training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. If true, units that reference the future are masked.
        scope: Optional scope for `variable_scope`.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        d_model = queries.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
            K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
            V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

            # Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

            outputs += queries

        return outputs

    def deepnn(self, x1, x2, x3, y, mask, padding, seq_length, keep_prob, is_train):  # deep CNN

        with tf.name_scope('PSSMPSS'):  # average

            embeddings = tf.keras.layers.Embedding(hc_ss_feature_size, hc_embedding_size, mask_zero=True)
            x2 = embeddings(x2)
            x = tf.concat([x1, x2], axis=2)

            x = self.double_LSTM(x, seq_length)
            x = tf.keras.layers.LayerNormalization(axis=2)(x)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.multiply(x, mask)


            x = self.multihead_attention(x, x, x, padding, hc_attention_head_number, keep_prob, is_train, False, "scaled_dot_product_attention")
            x = tf.keras.layers.LayerNormalization(axis=2)(x)
            x = tf.nn.dropout(x, keep_prob)

            mask_sum = tf.reduce_sum(mask, axis=1)
            x = tf.multiply(x, mask)
            x = tf.reduce_sum(x, axis=1) / mask_sum

            if(self.go_type == "CC"):
                x = tf.layers.dense(inputs=x, units=hc_full_connect_number, activation=None, use_bias=True)
            else:
                x = tf.layers.dense(inputs=x, units=hc_full_connect_number, activation=tf.nn.relu, use_bias=True)

            x = tf.nn.dropout(x, keep_prob)

        with tf.name_scope('InterPro'):

            if(self.go_type == "CC"):
                x3 = tf.layers.dense(inputs=x3, units=hc_interpro_layer_fc_number[self.go_type], activation=None, use_bias=True)
            else:
                x3 = tf.layers.dense(inputs=x3, units=hc_interpro_layer_fc_number[self.go_type], activation=tf.nn.relu, use_bias=True)

            x3 = tf.nn.dropout(x3, keep_prob)

        with tf.name_scope('Combine'):
            x = tf.concat([x, x3], axis=1)
            x = tf.keras.layers.LayerNormalization(axis=1)(x)
            x = tf.layers.dense(inputs=x, units=hc_full_connect_number, activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)

        with tf.name_scope('embeddings'):
            embeddings = tf.nn.l2_normalize(x, axis=1)

        with tf.name_scope('output'):  # output layer
            y_pred = tf.layers.dense(inputs=x, units=go_label_size[self.go_type], activation=tf.nn.sigmoid,
                                     name="output")

        with tf.name_scope('caculate_loss'):  # loss function

            triplet_loss = tl.batch_hard_triplet_loss(embeddings, y, hc_t_cut_off, hc_t_margin)

            cross_entropy = y * tf.log(y_pred + 1e-6) + (1 - y) * tf.log(1 + 1e-6 - y_pred)
            cross_entropy = -tf.reduce_mean(cross_entropy)

            triplet_loss = hc_alpha * triplet_loss + cross_entropy

        with tf.name_scope('adam_optimizer'):  # optimization

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(hc_learning_rate).minimize(triplet_loss)

        return train_step, y_pred, embeddings, cross_entropy

    def running(self):  # running process

        tf.reset_default_graph()
        tf.global_variables_initializer()

        x1 = tf.placeholder(tf.float32, [None, sequence_cut_off, hc_pssm_feature_size])
        x2 = tf.placeholder(tf.float32, [None, sequence_cut_off])
        x3 = tf.placeholder(tf.float32, [None, hc_interpro_feature_size])
        y_ = tf.placeholder(tf.float32, [None, go_label_size[self.go_type]])
        mask = tf.placeholder(tf.float32, [None, sequence_cut_off, hc_mask_dim])
        padding = tf.placeholder(tf.float32, [None, sequence_cut_off])
        seq_length = tf.placeholder(tf.int32, [None])
        keep_prob = tf.placeholder(tf.float32)
        is_train = tf.placeholder(tf.bool)

        train_step, probability, embeddings, cross_entropy = self.deepnn(x1, x2, x3, y_, mask, padding, seq_length,
                                                                         keep_prob, is_train)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = device_type_dict["gpu"]["gpu_ratio"]

        with tf.Session(config=config) as sess:

            ckpt = tf.train.latest_checkpoint(self.model_dir)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)
            print("successful load")


            '''
            train_data_list = rd.create_batch(self.train_feature_array1, self.train_feature_array2,
                                              self.train_feature_array3, self.train_label_array, self.train_name_list,
                                              hc_batch_size, True)
            '''

            test_data_list = rd.create_batch(self.test_feature_array1, self.test_feature_array2,
                                             self.test_feature_array3, self.test_label_array, self.test_name_list,
                                             hc_batch_size, False)

            '''
            # training
            train_loss = 0
            i = 0
            train_output = np.zeros([1, hc_full_connect_number])
            train_name_list = []

            for train_data in train_data_list:
                start_time = time.time()

                train_feature1, train_feature2, train_feature3, train_label, train_name = train_data
                train_length, train_mask, train_padding = rd.create_padding(train_name, self.train_sequence_length_dict,
                                                                            hc_mask_dim)
                current_loss = sess.run(cross_entropy,
                                        feed_dict={x1: train_feature1, x2: train_feature2, x3: train_feature3,
                                                   y_: train_label, mask: train_mask, padding: train_padding,
                                                   seq_length: train_length, keep_prob: hc_drop_prob, is_train: True})
                train_loss = train_loss + current_loss
                current_train_output = sess.run(embeddings,
                                                feed_dict={x1: train_feature1, x2: train_feature2, x3: train_feature3,
                                                           y_: train_label, mask: train_mask, padding: train_padding,
                                                           seq_length: train_length, keep_prob: hc_drop_prob,
                                                           is_train: True})
                train_output = np.concatenate((train_output, current_train_output), axis=0)
                train_name_list.extend(train_name)

                end_time = time.time()
                run_time = end_time - start_time
                i = i + 1
                print("The " + str(i) + "-th training bacth in all " + str(
                    len(train_data_list)) + " batchs, training loss = " + str(
                    current_loss) + ", running time is " + str(run_time * 1000) + " ms")


            train_output = train_output[1:, :]

            print("All training loss=" + str(train_loss) + "\n")
            '''

            #np.save(self.train_embedding_file, train_output)
            print("Loading training embeddings....")
            train_output = np.load(self.train_embedding_file)

            # testing
            test_loss = 0
            test_name_list = []
            test_pred_matrix = np.zeros([1, go_label_size[self.go_type]])
            test_output = np.zeros([1, hc_full_connect_number])

            for test_data in test_data_list:
                test_feature1, test_feature2, test_feature3, test_label, test_name = test_data
                test_length, test_mask, test_padding = rd.create_padding(test_name, self.test_sequence_length_dict, hc_mask_dim)

                test_loss = test_loss + sess.run(cross_entropy,
                                                 feed_dict={x1: test_feature1, x2: test_feature2, x3: test_feature3,
                                                            y_: test_label, mask: test_mask, padding: test_padding,
                                                            seq_length: test_length, keep_prob: 1.0, is_train: False})
                current_y_pred = sess.run(probability,
                                          feed_dict={x1: test_feature1, x2: test_feature2, x3: test_feature3,
                                                     y_: test_label, mask: test_mask, padding: test_padding,
                                                     seq_length: test_length, keep_prob: 1.0, is_train: False})
                current_test_output = sess.run(embeddings,
                                               feed_dict={x1: test_feature1, x2: test_feature2, x3: test_feature3,
                                                          y_: test_label, mask: test_mask, padding: test_padding,
                                                          seq_length: test_length, keep_prob: 1.0, is_train: False})

                test_name_list.extend(test_name)
                test_pred_matrix = np.concatenate((test_pred_matrix, current_y_pred), axis=0)
                test_output = np.concatenate((test_output, current_test_output), axis=0)

            test_loss = test_loss / len(test_name_list)
            test_pred_matrix = test_pred_matrix[1:]
            test_output = test_output[1:, :]


            sh.save_cross_entropy_results(self.cross_entropy_result_dir, self.go_type, self.term_list_file, hc_cross_entropy_name, test_name_list, test_pred_matrix)
            sh.save_distance(self.distance_dir, train_output, test_output, self.train_name_list, test_name_list)


if __name__ == '__main__':

    workdir = sys.argv[1]
    go_type = "MF"
    round_index = 1

    lstm = LSTM(workdir, go_type, round_index)
    lstm.running()




