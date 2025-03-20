import tensorflow as tf
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def pairwise_distances(embeddings):      # calculate the Euclidean distance

    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    distances = tf.maximum(distances, 0.0)

    mask = tf.to_float(tf.equal(distances, 0.0))
    distances = distances + mask * 1e-16
    distances = tf.sqrt(distances)
    distances = distances * (1.0 - mask)

    return distances


def pairwise_cos(embeddings):      # calculate the Euclidean distance

    distances  = tf.matmul(embeddings, tf.transpose(embeddings))

    return distances


def create_f1_score_matrixs(labels):  # calculate the f1-score between two genes

    tp_matrix = tf.matmul(labels, tf.transpose(labels))
    sum_matrix = tf.reduce_sum(labels, axis=1)

    m_matrix = tf.expand_dims(sum_matrix, 1)
    precision_matrix = tp_matrix / m_matrix

    mask = tf.to_float(tf.equal(precision_matrix, 0.0))
    precision_matrix = precision_matrix + mask * 1e-16

    n_matrix = tf.expand_dims(sum_matrix, 0)
    recall_matrix = tp_matrix / n_matrix

    f1_score_matrix = 2 * tf.multiply(precision_matrix, recall_matrix) / (precision_matrix + recall_matrix)

    return f1_score_matrix

def get_triplet_mask(f1_score_matrix, cut_off):  # get valid triplet index matrix

    sample_number = tf.shape(f1_score_matrix)[0]

    indices_equal = tf.cast(tf.eye(sample_number), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    ones = tf.ones_like(f1_score_matrix)
    zeros = tf.zeros_like(f1_score_matrix)
    pos_matrix = tf.where(f1_score_matrix > cut_off, ones, zeros)
    label_equal = tf.cast(pos_matrix, tf.bool)

    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)
    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    triplet_index = tf.logical_and(distinct_indices, valid_labels)

    return triplet_index

def batch_all_triplet_loss(embeddings, labels, cut_off, margin):  # batch on all triplets

    pairwise_dist = pairwise_distances(embeddings)
    f1_score_matrix = create_f1_score_matrixs(labels)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)

    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = get_triplet_mask(f1_score_matrix, cut_off)
    mask = tf.to_float(mask)

    triplet_loss = tf.multiply(mask, triplet_loss)

    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    #fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss

def get_anchor_positive_triplet_mask(f1_score_matrix, cut_off):  # get positive index matrix

    sample_number = tf.shape(f1_score_matrix)[0]
    indices_equal = tf.cast(tf.eye(sample_number), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    ones = tf.ones_like(f1_score_matrix)
    zeros = tf.zeros_like(f1_score_matrix)
    pos_matrix = tf.where(f1_score_matrix >= cut_off, ones, zeros)
    pos_matrix = tf.cast(pos_matrix, tf.bool)

    pos_matrix = tf.logical_and(indices_not_equal, pos_matrix)

    return pos_matrix

def get_anchor_negative_triplet_mask(f1_score_matrix, cut_off):  # get negative index matrix

    ones = tf.ones_like(f1_score_matrix)
    zeros = tf.zeros_like(f1_score_matrix)
    neg_matrix = tf.where(f1_score_matrix < cut_off, ones, zeros)
    neg_matrix = tf.cast(neg_matrix, tf.bool)

    sample_number = tf.shape(f1_score_matrix)[0]
    indices_equal = tf.cast(tf.eye(sample_number), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    neg_matrix = tf.logical_and(indices_not_equal, neg_matrix)

    return neg_matrix


def batch_hard_triplet_loss(embeddings, labels, cut_off, margin):   # batch on hard triplets

    pairwise_dist = pairwise_distances(embeddings)
    pairwise_dist = tf.square(pairwise_dist) / 4.0

    #pairwise_dist = pairwise_cos(embeddings)

    f1_score_matrix = create_f1_score_matrixs(labels)
    mask_anchor_positive = get_anchor_positive_triplet_mask(f1_score_matrix, cut_off)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    mask_anchor_negative = get_anchor_negative_triplet_mask(f1_score_matrix, cut_off)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    triplet_loss = tf.reduce_mean(triplet_loss)


    return triplet_loss



def global_loss(embeddings, labels, cut_off, margin, lamda):  # calculate global loss

    pairwise_dist = pairwise_distances(embeddings)
    pairwise_dist = tf.square(pairwise_dist) / 4.0

    f1_score_matrix = create_f1_score_matrixs(labels)

    mask_anchor_positive = get_anchor_positive_triplet_mask(f1_score_matrix, cut_off)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    mask_anchor_negative = get_anchor_negative_triplet_mask(f1_score_matrix, cut_off)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)

    num_positive = tf.reduce_sum(mask_anchor_positive)
    num_negative = tf.reduce_sum(mask_anchor_negative)

    positive_mean = tf.reduce_sum(positive_dist)/(num_positive+1e-16)
    negative_mean = tf.reduce_sum(negative_dist)/(num_negative+1e-16)

    positive_var = positive_dist - positive_mean
    negative_var = negative_dist - negative_mean

    positive_var = tf.square(positive_var)
    positive_var = tf.multiply(mask_anchor_positive, positive_var)
    positive_var = tf.reduce_sum(positive_var)/(num_positive+1e-16)

    negative_var = tf.square(negative_var)
    negative_var = tf.multiply(mask_anchor_negative, negative_var)
    negative_var = tf.reduce_sum(negative_var)/(num_negative+1e-16)

    part1 = positive_var + negative_var


    return part1





