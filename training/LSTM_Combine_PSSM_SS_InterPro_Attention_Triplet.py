import sys
import os
import Read_Data_Combine_PSSM_SS_InterPro_Attention as rd
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
gpu_ratio = float(sys.argv[4])
import Calculate_results as cr
import Triplet_Loss as tl

import time

class LSTM(object):

    def __init__(self,
                 workdir,
                 type,
                 feature_dir1,
                 feature_dir2,
                 feature_dir3,
                 sequence_length_file,
                 max_length,
                 feature_size1,
                 feature_size2,
                 feature_size3,
                 label_size,
                 batch_size,
                 mask_dim,
                 rnn_unit,
                 full_connect_number,
                 drop_prob,
                 learning_rate,
                 max_iteration,
                 round_index):

        self.workdir = workdir
        self.type = type
        self.feature_dir1 = feature_dir1
        self.feature_dir2 = feature_dir2
        self.feature_dir3 = feature_dir3
        self.sequence_length_file = sequence_length_file
        self.max_length = max_length
        self.feature_size1 = feature_size1
        self.feature_size2 = feature_size2
        self.feature_size3 = feature_size3
        self.label_size = label_size
        self.batch_size = batch_size
        self.mask_dim = mask_dim
        self.rnn_unit = rnn_unit
        self.full_connect_number = full_connect_number
        self.drop_prob = drop_prob
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.round_index = round_index

        self.model_dir = workdir + "/" + type + "/model/" + str(round_index) + "/"
        os.system("rm -rf " + self.model_dir)
        os.makedirs(self.model_dir)

        self.recordfile = workdir + "/" + type + "/record_loss" + str(round_index)


        self.train_feature_array1, self.train_feature_array2, self.train_feature_array3, self.train_label_array, self.train_name_list, self.sequence_length_dict = rd.read_data(
            workdir, feature_dir1, feature_dir2, feature_dir3, sequence_length_file, type, "train")
        self.evaluate_feature_array1, self.evaluate_feature_array2, self.evaluate_feature_array3, self.evaluate_label_array, self.evaluate_name_list, self.sequence_length_dict = rd.read_data(
            workdir, feature_dir1, feature_dir2, feature_dir3, sequence_length_file, type, "evaluate")
        self.test_feature_array1, self.test_feature_array2, self.test_feature_array3, self.test_label_array, self.test_name_list, self.sequence_length_dict = rd.read_data(
            workdir, feature_dir1, feature_dir2, feature_dir3, sequence_length_file, type, "test")


    def double_LSTM(self, x, seq_length, keep_prob):  # double direction LSTM

        cell_fw_lstm_cells = tf.nn.rnn_cell.LSTMCell(self.rnn_unit, name = "cell1")
        cell_bw_lstm_cells = tf.nn.rnn_cell.LSTMCell(self.rnn_unit, name = "cell2")

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

            embeddings = tf.keras.layers.Embedding(self.feature_size2, 128, mask_zero=True)
            x2 = embeddings(x2)
            x = tf.concat([x1, x2], axis=2)

            x = self.double_LSTM(x, seq_length, keep_prob)
            x = tf.keras.layers.LayerNormalization(axis=2)(x)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.multiply(x, mask)

            x = self.multihead_attention(x, x, x, padding, 8, keep_prob, is_train, False, "scaled_dot_product_attention")
            x = tf.keras.layers.LayerNormalization(axis=2)(x)
            x = tf.nn.dropout(x, keep_prob)

            mask_sum = tf.reduce_sum(mask, axis=1)
            x = tf.multiply(x, mask)
            x = tf.reduce_sum(x, axis=1) / mask_sum

            x = tf.layers.dense(inputs=x, units=self.full_connect_number, activation = tf.nn.relu, use_bias=True)
            x = tf.nn.dropout(x, keep_prob)

        with tf.name_scope('InterPro'):

            x3 = tf.layers.dense(inputs=x3, units=self.full_connect_number, activation = tf.nn.relu, use_bias=True)
            x3 = tf.nn.dropout(x3, keep_prob)

        with tf.name_scope('Combine'):

            x = tf.concat([x, x3], axis=1)
            x = tf.keras.layers.LayerNormalization(axis=1)(x)
            x = tf.layers.dense(inputs=x, units=self.full_connect_number, activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)

        with tf.name_scope('embeddings'):

            embeddings = tf.nn.l2_normalize(x, axis=1)

        with tf.name_scope('output'):  # output layer

            y_pred = tf.layers.dense(inputs=x, units=self.label_size, activation=tf.nn.sigmoid, name = "output")

        with tf.name_scope('caculate_loss'):   # loss function

            t_cut_off = 0.8
            t_margin = 0.1
            alpha = 0.1

            triplet_loss = tl.batch_hard_triplet_loss(embeddings, y, t_cut_off, t_margin)

            cross_entropy = y * tf.log(y_pred + 1e-6) + (1 - y) * tf.log(1 + 1e-6 - y_pred)
            cross_entropy = -tf.reduce_mean(cross_entropy)

            triplet_loss = alpha * triplet_loss + cross_entropy

        with tf.name_scope('adam_optimizer'):  # optimization

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(triplet_loss)

        return train_step, y_pred, embeddings, triplet_loss

    def running(self): #running process

        tf.reset_default_graph()
        tf.global_variables_initializer()

        x1 = tf.placeholder(tf.float32, [None, self.max_length, self.feature_size1])
        x2 = tf.placeholder(tf.float32, [None, self.max_length])
        x3 = tf.placeholder(tf.float32, [None, self.feature_size3])
        y_ = tf.placeholder(tf.float32, [None, self.label_size])
        mask = tf.placeholder(tf.float32, [None, self.max_length, self.mask_dim])
        padding = tf.placeholder(tf.float32, [None, self.max_length])
        seq_length = tf.placeholder(tf.int32, [None])
        keep_prob = tf.placeholder(tf.float32)
        is_train = tf.placeholder(tf.bool)

        train_step, probability, embeddings, cross_entropy = self.deepnn(x1, x2, x3, y_, mask, padding, seq_length, keep_prob, is_train)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_ratio

        with tf.Session(config=config) as sess:

            if (os.path.exists(self.model_dir + "/checkpoint")):

                ckpt = tf.train.latest_checkpoint(self.model_dir)
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
                print("successful load")

            else:
                sess.run(tf.global_variables_initializer())

            f = open(self.recordfile, "w")

            for iteration in range(1, self.max_iteration + 1):


                train_data_list = rd.create_batch(self.train_feature_array1, self.train_feature_array2, self.train_feature_array3, self.train_label_array, self.train_name_list,
                                                  self.batch_size, True)
                evaluate_data_list = rd.create_batch(self.evaluate_feature_array1, self.evaluate_feature_array2, self.evaluate_feature_array3, self.evaluate_label_array, self.evaluate_name_list,
                                                  self.batch_size, False)
                test_data_list = rd.create_batch(self.test_feature_array1, self.test_feature_array2, self.test_feature_array3, self.test_label_array, self.test_name_list,
                                                  self.batch_size, False)


                # training
                train_loss = 0

                i = 0
                #train_output = np.zeros([1, self.full_connect_number])
                train_name_list = []

                for train_data in train_data_list:

                    start_time = time.time()

                    train_feature1, train_feature2, train_feature3, train_label, train_name = train_data
                    train_length, train_mask, train_padding = rd.create_padding(train_name, self.sequence_length_dict, self.mask_dim)

                    train_step.run(feed_dict={x1: train_feature1, x2: train_feature2, x3: train_feature3, y_: train_label, mask: train_mask, padding: train_padding, seq_length: train_length, keep_prob: self.drop_prob, is_train: True})
                    current_loss = sess.run(cross_entropy, feed_dict={x1: train_feature1, x2: train_feature2, x3: train_feature3, y_: train_label, mask: train_mask, padding: train_padding, seq_length: train_length, keep_prob: self.drop_prob, is_train: True})
                    train_loss = train_loss + current_loss

                    #current_train_output = sess.run(embeddings, feed_dict={x1: train_feature1, x2: train_feature2, x3: train_feature3, y_: train_label, mask: train_mask, padding: train_padding, seq_length: train_length, keep_prob: self.drop_prob, is_train: True})
                    #train_output = np.concatenate((train_output, current_train_output), axis=0)
                    train_name_list.extend(train_name)

                    end_time = time.time()
                    run_time = end_time - start_time
                    i = i + 1
                    print("The " + str(i) + "-th training bacth in all " + str(len(train_data_list)) + " batchs, training loss = " + str(current_loss) + ", running time is " + str(run_time*1000) + " ms")

                    #del train_data

                #train_output = train_output[1:, :]


                f.write("The " + str(iteration) + "-th iteration: \ntraining loss=" + str(train_loss) + "\n")
                f.flush()

                # save model
                saver = tf.train.Saver()
                saver.save(sess, self.model_dir + "model" + str(iteration))

                # validation

                evaluate_loss  = 0
                evaluate_name_list = []
                evaluate_pred_matrix = np.zeros([1, self.label_size])
                #evaluate_output = np.zeros([1, full_connect_number])

                for evaluate_data in evaluate_data_list:

                    evaluate_feature1, evaluate_feature2, evaluate_feature3, evaluate_label, evaluate_name = evaluate_data
                    evaluate_length, evaluate_mask, evaluate_padding = rd.create_padding(evaluate_name, self.sequence_length_dict, self.mask_dim)

                    evaluate_loss = evaluate_loss + sess.run(cross_entropy, feed_dict={x1: evaluate_feature1, x2: evaluate_feature2, x3: evaluate_feature3, y_: evaluate_label, mask: evaluate_mask, padding: evaluate_padding, seq_length: evaluate_length, keep_prob: 1.0, is_train: False})
                    current_y_pred = sess.run(probability, feed_dict={x1: evaluate_feature1, x2: evaluate_feature2, x3: evaluate_feature3, y_: evaluate_label, mask: evaluate_mask, padding: evaluate_padding, seq_length: evaluate_length, keep_prob: 1.0, is_train: False})
                    #current_evaluate_output = sess.run(embeddings, feed_dict={x1: evaluate_feature1, x2: evaluate_feature2, x3: evaluate_feature3, y_: evaluate_label, mask: evaluate_mask, padding: evaluate_padding, seq_length: evaluate_length, keep_prob: 1.0, is_train: False})

                    evaluate_name_list.extend(evaluate_name)
                    evaluate_pred_matrix = np.concatenate((evaluate_pred_matrix, current_y_pred), axis=0)
                    #evaluate_output = np.concatenate((evaluate_output, current_evaluate_output), axis=0)

                evaluate_loss = evaluate_loss / len(evaluate_name_list)
                evaluate_pred_matrix = evaluate_pred_matrix[1:]
                #evaluate_output = evaluate_output[1:, :]

                f.write("validation loss=" + str(evaluate_loss) + "\n")
                f.flush()


                # testing
                test_loss = 0
                test_name_list = []
                test_pred_matrix = np.zeros([1, self.label_size])
                #test_output = np.zeros([1, full_connect_number])

                for test_data in test_data_list:

                    test_feature1, test_feature2, test_feature3, test_label, test_name = test_data
                    test_length, test_mask, test_padding = rd.create_padding(test_name, self.sequence_length_dict, self.mask_dim)

                    test_loss = test_loss + sess.run(cross_entropy, feed_dict={x1: test_feature1, x2: test_feature2, x3: test_feature3, y_: test_label, mask: test_mask, padding: test_padding, seq_length: test_length, keep_prob: 1.0, is_train: False})
                    current_y_pred = sess.run(probability, feed_dict={x1: test_feature1, x2: test_feature2, x3: test_feature3, y_: test_label, mask: test_mask, padding: test_padding, seq_length: test_length, keep_prob: 1.0, is_train: False})
                    #current_test_output = sess.run(embeddings, feed_dict={x1: test_feature1, x2: test_feature2, x3: test_feature3, y_: test_label, mask: test_mask, padding: test_padding, seq_length: test_length, keep_prob: 1.0, is_train: False})

                    test_name_list.extend(test_name)
                    test_pred_matrix = np.concatenate((test_pred_matrix, current_y_pred), axis=0)
                    #test_output = np.concatenate((test_output, current_test_output), axis=0)

                test_loss = test_loss / len(test_name_list)
                test_pred_matrix = test_pred_matrix[1:]
                #test_output = test_output[1:, :]

                f.write("testing loss=" + str(test_loss) + "\n")
                f.flush()

                # evaluation results
                cr.calculate_pred_label(workdir, self.type, evaluate_name_list, evaluate_pred_matrix, test_name_list, test_pred_matrix, self.round_index, iteration)

            f.close()

if __name__ == '__main__':

    workdir = sys.argv[1]
    type = sys.argv[2]

    feature_dir1 = "/data/yihengzhu/GOA/resource/pssm_uniprot_array/"
    feature_dir2 = "/data/yihengzhu/GOA/resource/ss_index/"
    feature_dir3 = "/data/yihengzhu/toolbars/sequence_homology_tools/InterPro/temps/entry_array/"
    sequence_length_file = "/data/yihengzhu/GOA/resource/all_protein_length"

    max_length = 1024
    feature_size1 = 20
    feature_size2 = 10
    feature_size3 = 45899
    label_size = 6858
    batch_size = 128
    mask_dim = 256
    rnn_unit = 128
    full_connect_number = 1024
    drop_prob = 0.8
    learning_rate = 0.0001
    max_iteration = 30
    for round_index in range(1, 6):
        lstm = LSTM(workdir, type, feature_dir1, feature_dir2, feature_dir3, sequence_length_file, max_length,
                    feature_size1, feature_size2, feature_size3, label_size, batch_size, mask_dim, rnn_unit,
                    full_connect_number, drop_prob, learning_rate, max_iteration, round_index)
        lstm.running()








