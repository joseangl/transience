#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras


def lrelu(x):
    return keras.activations.relu(x, alpha=0.03)  # See Trigeorgis et al (2018)


def get_activation(activation):
    if activation == 'lrelu':
        act_fun = lrelu
    else:
        act_fun = keras.activations.get(activation)
    return act_fun


def inverse_root_matrix(x):
    pow_value = tf.constant([-0.5])
    d, v = tf.linalg.eigh(x)
    return tf.matmul(tf.matmul(v, tf.linalg.diag(tf.pow(d, pow_value))), v, transpose_b=True)
    # for stability
    # D1_indices = tf.where(D1 > eps)
    # D1_indices = tf.squeeze(D1_indices)
    # V1 = tf.gather(V1, D1_indices)
    # D1 = tf.gather(D1, D1_indices)
    #
    # D2_indices = tf.where(D2 > eps)
    # D2_indices = tf.squeeze(D2_indices)
    # V2 = tf.gather(V2, D2_indices)
    # D2 = tf.gather(D2, D2_indices)
    # eps = tf.constant([1e-12])
    # s, u, v = tf.svd(x, full_matrices=True)
    # return tf.matmul(tf.matmul(u, tf.diag(tf.pow(s, pow_value))), v, transpose_b=True)
    # x_root = tf.cholesky(x)
    # return tf.matrix_inverse(x_root)


def cca_loss(H1, H2):
    """
    It is the loss function of CCA as introduced in the original paper. There can be other formulations.
    It is implemented by Tensorflow tensor operations.
    """
    # reg_val = tf.constant([1e-4])
    dim = tf.shape(H1)[1]
    reg_val = tf.linalg.diag(tf.ones(dim) * tf.constant(1e-4))
    # dim = tf.shape(y_pred)[1] // 2

    # unpack (separate) the output of networks for view 1 and view 2
    # H1 = y_pred[:, 0:dim]
    # H2 = y_pred[:, dim:]
    num_samples = tf.cast(tf.shape(H1)[0], tf.float32)

    # subtract the mean value
    H1bar = H1 - tf.reduce_mean(H1, 0, keepdims=True)
    H2bar = H2 - tf.reduce_mean(H2, 0, keepdims=True)

    # calculate the auto-covariance and cross-covariance
    SigmaHat12 = tf.divide(tf.matmul(H1bar, H2bar, transpose_a=True), num_samples - 1)
    SigmaHat11 = tf.divide(tf.matmul(H1bar, H1bar, transpose_a=True), num_samples - 1) + reg_val
    SigmaHat22 = tf.divide(tf.matmul(H2bar, H2bar, transpose_a=True), num_samples - 1) + reg_val

    # calculate the root inverse of covariance matrices
    SigmaHat11RootInv = inverse_root_matrix(SigmaHat11)
    SigmaHat22RootInv = inverse_root_matrix(SigmaHat22)

    # The correlation is defined as the nuclear norm of matrix T (i.e. sqrt(trace(T'*T))
    T = tf.matmul(SigmaHat11RootInv, tf.matmul(SigmaHat12, SigmaHat22RootInv))
    return -tf.sqrt(tf.linalg.trace(tf.matmul(T, T, transpose_a=True)))


def kde(X, sigma):
    dim = tf.shape(X)[1]
    num_samples = tf.shape(X)[0]
    mixture_weights = tfd.Categorical(probs=tf.ones(num_samples) / tf.cast(num_samples, tf.float32))
    return tfd.MixtureSameFamily(mixture_distribution=mixture_weights,
                                 components_distribution=tfd.MultivariateNormalDiag(loc=X, scale_diag=tf.ones(dim) * sigma))


# def mutual_information(index, latent_joint, latent_x, latent_y, mvn_joint, mvn_x, mvn_y):
def mutual_information(index, latent_joint, latent_x, latent_y, sigma_joint, sigma_x, sigma_y):
    num_samples = tf.shape(latent_joint)[0]
    range_ = tf.range(num_samples, dtype=tf.int32)
    mask = tf.math.not_equal(range_, index)

    p_joint = kde(tf.boolean_mask(latent_joint, mask), sigma_joint)
    p_x = kde(tf.boolean_mask(latent_x, mask), sigma_x)
    p_y = kde(tf.boolean_mask(latent_y, mask), sigma_y)
    log_prob_joint = p_joint.log_prob(latent_joint[index])
    return tf.math.exp(log_prob_joint) * (log_prob_joint - (p_x.log_prob(latent_x[index]) + p_y.log_prob(latent_y[index])))
    # norm_factor = tf.cast(num_samples - 1, tf.float32)
    # p_joint = tf.reduce_sum(tf.boolean_mask(mvn_joint.prob(latent_joint[index]), mask)) / norm_factor
    # p_x = tf.reduce_sum(tf.boolean_mask(mvn_x.prob(latent_x[index]), mask)) / norm_factor
    # p_y = tf.reduce_sum(tf.boolean_mask(mvn_y.prob(latent_y[index]), mask)) / norm_factor
    # return p_joint * (tf.math.log(p_joint) - (tf.math.log(p_x) + tf.math.log(p_y)))


def kl_loss_standard_normal(x):
    '''
    See derivations in:
    (1) https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
    (2) https://en.wikipedia.org/wiki/Relative_entropy#Examples
    '''
    # Estimate the mean and variances of the data
    reg_val = tf.constant(1e-6)
    dim = tf.cast(tf.shape(x)[1], dtype=tf.float32)
    mu2 = tf.square(tf.math.reduce_mean(x, axis=0))
    var = tf.math.reduce_variance(x, axis=0)
    return tf.constant(0.5) * (tf.reduce_sum(var + mu2 - tf.math.log(var + reg_val)) - dim)


def contrastive_loss(latent_x, latent_y, margin):
    d_positive = tf.constant(1.0) + keras.losses.cosine_similarity(latent_x, latent_y)
    # d_positive = tf.reduce_sum((latent_x - latent_y)**2, axis=-1)
    # d_negative = tf.constant(1.0) + keras.losses.cosine_similarity(latent_x, tf.random.shuffle(latent_y))  # Tensorflow complaints with random shuffle. See. https://github.com/tensorflow/tensorflow/issues/6269
    latent_y_shuffled = tf.gather(latent_y, tf.random.shuffle(tf.range(tf.shape(latent_y)[0])))
    d_negative = tf.constant(1.0) + keras.losses.cosine_similarity(latent_x, latent_y_shuffled)
    # d_negative = tf.reduce_sum((latent_x - latent_y_shuffled)**2, axis=-1)
    return tf.math.reduce_mean(tf.math.maximum(d_positive - d_negative + margin, 0.0))


# def mmi_loss(y_pred, sigma_joint, sigma_x, sigma_y):
# def mmi_loss(latent_x, latent_y, sigma_val=1.0):
#     """"
#     Maximal mutual information loss fuction.
#     y_true is ignored.
#     """
#     # dim = tf.shape(y_pred)[1] // 2
#     # dim = tf.shape(latent_x)[1]
#     # sigma = tf.constant(sigma_val)

#     # unpack (separate) the output of networks for view 1 and view 2
#     # latent_joint = y_pred
#     # latent_x = y_pred[:, 0:dim]
#     # latent_y = y_pred[:, dim:]
#     # latent_joint = tf.concat([latent_x, latent_y], axis=1)
#     # num_samples = tf.shape(latent_joint)[0]
#     # num_samples_float = tf.cast(num_samples, tf.float32)

#     # mixture_weights = tfd.Categorical(probs=tf.ones(num_samples) / num_samples_float)
#     # p_joint = tfd.MixtureSameFamily(
#     #     mixture_distribution=mixture_weights,
#     #     components_distribution=tfd.MultivariateNormalDiag(loc=latent_joint, scale_diag=tf.ones(2 * dim) * sigma))
#     # p_x = tfd.MixtureSameFamily(
#     #     mixture_distribution=mixture_weights,
#     #     components_distribution=tfd.MultivariateNormalDiag(loc=latent_x, scale_diag=tf.ones(dim) * sigma))
#     # p_y = tfd.MixtureSameFamily(
#     #     mixture_distribution=mixture_weights,
#     #     components_distribution=tfd.MultivariateNormalDiag(loc=latent_y, scale_diag=tf.ones(dim) * sigma))

#     # # Computation of the inverse mutual information
#     # log_prob_joint = p_joint.log_prob(latent_joint)
#     # return -tf.reduce_sum(tf.math.exp(log_prob_joint) * (log_prob_joint - (p_x.log_prob(latent_x) + p_y.log_prob(latent_y))))
#
#     latent_joint = tf.concat([latent_x, latent_y], axis=1)
#     num_samples = tf.shape(latent_joint)[0]
#     # KDE distributions
#     # I_xy = tf.map_fn(fn=lambda x: mutual_information(x[0], x[1], latent_joint, sigma_val),
#     #                  elems=(latent_joint, tf.expand_dims(tf.range(num_samples), 1)), dtype=tf.float32)
#     # I_xy = keras.backend.map_fn(lambda x: mutual_information(x, latent_joint, sigma_val), tf.range(num_samples))
#     I_xy = tf.stack([mutual_information(x, i, latent_joint, sigma_val) for i, x in enumerate(tf.unstack(latent_joint))])
#     return -tf.reduce_sum(I_xy)


def predict_variables(model, dataset):
    # Predict the latent representations
    if isinstance(dataset, list):
        return [model.predict(x) for x in dataset]
    else:
        return model.predict(dataset)


class KLLossLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(KLLossLayer, self).__init__(**kwargs)

    def call(self, input, training=None):
        if training:
            self.add_loss(kl_loss_standard_normal(input))
        return input


class CCALossLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CCALossLayer, self).__init__(**kwargs)

    def call(self, latent1, latent2, training=None):
        if training:
            self.add_loss(cca_loss(latent1, latent2))
        return (latent1, latent2)


class MMILossLayer(keras.layers.Layer):
    def __init__(self, sigma_joint=1.0, sigma_x=1.0, sigma_y=1.0, self_probs=True, **kwargs):
        super(MMILossLayer, self).__init__(**kwargs)
        # Trainable bandwiths for the Gaussian kernerls used in KDE
        self._sigma_joint = tf.Variable(sigma_joint, constraint=keras.constraints.NonNeg(), dtype=tf.float32, trainable=True, name='sigma_joint')
        self._sigma_x = tf.Variable(sigma_x, constraint=keras.constraints.NonNeg(), dtype=tf.float32, trainable=True, name='sigma_x')
        self._sigma_y = tf.Variable(sigma_y, constraint=keras.constraints.NonNeg(), dtype=tf.float32, trainable=True, name='sigma_y')
        self._compute_self_probs = self_probs

    def get_config(self):
        return {"sigma_joint": self._sigma_joint.numpy(),
                "sigma_x": self._sigma_x.numpy(),
                "sigma_y": self._sigma_y.numpy(),
                "self_probs": self._compute_self_probs}  # To make the custom layer serializable

    def get_vars(self):
        return self._sigma_joint, self._sigma_x, self._sigma_y

    def custom_loss(self, latent_x, latent_y):
        latent_joint = tf.concat([latent_x, latent_y], axis=1)
        dim = tf.shape(latent_x)[1]
        num_samples = tf.shape(latent_joint)[0]

        # # KDE distributions
        # mvn_joint = tfd.MultivariateNormalDiag(loc=latent_joint, scale_diag=tf.ones(2 * dim) * self._sigma_joint)
        # mvn_x = tfd.MultivariateNormalDiag(loc=latent_x, scale_diag=tf.ones(dim) * self._sigma_x)
        # mvn_y = tfd.MultivariateNormalDiag(loc=latent_y, scale_diag=tf.ones(dim) * self._sigma_y)

        # Computation of the mutual information for each latent variable
        if self._compute_self_probs:
            p_joint = kde(latent_joint, self._sigma_joint)
            p_x = kde(latent_x, self._sigma_x)
            p_y = kde(latent_y, self._sigma_y)
            log_prob_joint = p_joint.log_prob(latent_joint)
            I_xy = tf.math.exp(log_prob_joint) * (log_prob_joint - (p_x.log_prob(latent_x) + p_y.log_prob(latent_y)))
        else:
            range_ = tf.range(num_samples, dtype=tf.int32)
            I_xy = tf.vectorized_map(fn=lambda elem_index: mutual_information(elem_index, latent_joint, latent_x, latent_y, self._sigma_joint, self._sigma_x, self._sigma_y),
                                     elems=range_)
            # I_xy = tf.vectorized_map(fn=lambda elem_index: mutual_information(elem_index, latent_joint, latent_x, latent_y, mvn_joint, mvn_x, mvn_y),
            #                          elems=range_)
        return -tf.reduce_sum(I_xy)

    def call(self, latent1, latent2, training=None):
        if training:
            self.add_loss(self.custom_loss(latent1, latent2))
        return (latent1, latent2)


# Implementation of the multiview contrastive loss described in, Wang et al, (2017). "Deep Variational Canonical Correlation Analysis"
class ContrastiveLossLayer(keras.layers.Layer):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLossLayer, self).__init__(**kwargs)
        self._margin = tf.constant(margin, dtype=tf.float32)

    def get_config(self):
        return {"margin": self._margin.numpy()}

    def call(self, latent1, latent2, training=None):
        if training:
            self.add_loss(contrastive_loss(latent1, latent2, self._margin))
        return (latent1, latent2)