#!/usr/bin/env python3

import time
import datetime
import os
import json
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd
from tensorflow import keras
import numpy as np
from librosa.sequence import dtw
import matplotlib.pyplot as plt
import librosa.display

# Constants
_EARLY_STOPPING_PATIENCE = 15
_DTW_BAND_RAD = 0.5

# Code execute every time the module is loaded
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def init_alignments(seq1, seq2):
    alignments = []
    for x, y in zip(seq1, seq2):
        # Uniform alignment between x and y
        n = max(x.shape[0], y.shape[0])
        path1 = np.linspace(0, x.shape[0] - 1, n).round().astype(int)
        path2 = np.linspace(0, y.shape[0] - 1, n).round().astype(int)
        alignments.append((path1, path2))
    return alignments


def compute_alignments(seq1, seq2, metric='euclidean', save_plots=False, filepath_prefix=None):
    alignments = []
    total_cost = 0
    for i, (x, y) in enumerate(zip(seq1, seq2)):
        # Align x and y using Dynamic Time Warping
        cost_matrix, path = dtw(x.T, y.T, metric=metric, backtrack=True, global_constraints=True, band_rad=_DTW_BAND_RAD)
        # cost_matrix, path = dtw(x.T, y.T, metric=metric, backtrack=True, global_constraints=False)
        total_cost += cost_matrix[-1, -1]
        alignments.append((np.flip(path[:, 0], axis=0), np.flip(path[:, 1], axis=0)))
        if save_plots and i % 50 == 0:
            librosa.display.specshow(cost_matrix, x_axis='frames', y_axis='frames')
            plt.title('DTW alignment')
            plt.plot(path[:, 1], path[:, 0], label='Optimal path', color='y')
            plt.savefig('{}_seq{}.pdf'.format(filepath_prefix, i))
            plt.clf()
    return alignments, total_cost


def align_data(train_set_1, train_set_2, alignments):
    train_set_1_aligned = []
    train_set_2_aligned = []
    for x, y, path in zip(train_set_1, train_set_2, alignments):
        train_set_1_aligned.append(x[path[0], :])
        train_set_2_aligned.append(y[path[1], :])
    return np.vstack(train_set_1_aligned), np.vstack(train_set_2_aligned)


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


def build_model(inputs, output_dim, network_arch, activation='relu', dropout=0.0, name=''):
    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5
    act_fun = get_activation(activation)

    if isinstance(inputs, int):
        input_layer = keras.layers.Input(shape=(inputs,), name='{}_input'.format(name))
    else:
        input_layer = inputs
    prev_layer = input_layer
    for layer, num_units in enumerate(network_arch):
        prev_layer = keras.layers.Dense(num_units, activation=act_fun, name='{}_hidden_{}'.format(name, layer))(prev_layer)
        # prev_layer = keras.layers.Dense(num_units, activation=act_fun, kernel_regularizer=keras.regularizers.l2(reg_par), name='{}_hidden_{}'.format(name, layer))(prev_layer)
        # lin_layer = keras.layers.Dense(num_units, name='{}_dense_{}'.format(model_name, layer))(prev_layer)
        # batch_norm_layer = keras.layers.BatchNormalization(name='{}_batchnorm_{}'.format(model_name, layer))(lin_layer)
        # act_layer = keras.layers.Activation(activation, name='{}_act_{}'.format(model_name, layer))(batch_norm_layer)
        if dropout > 0.0:
            prev_layer = keras.layers.Dropout(dropout, name='{}_dropout_{}'.format(name, layer))(prev_layer)
    # output_layer = keras.layers.Dense(output_dim, kernel_regularizer=keras.regularizers.l2(reg_par), name='{}_output'.format(name))(prev_layer)
    output_layer = keras.layers.Dense(output_dim, name='{}_output'.format(name))(prev_layer)
    return (input_layer, output_layer)


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


class MultiviewAligner(object):
    def __init__(self, verbose=True):
        self._verbose = verbose
        self._model = None
        self._dtw_distance = None
        self._model_ready = False
        np.random.seed(1234)
        tf.random.set_seed(1234)

    def build(self, view1_input_dim, view2_input_dim, latent_shared_dim, latent_private_dim, network_architecture, activation='relu', similarity='cca', cycle_loss=0.0, dropout=0.0, lr=1e-4):
        if self._verbose:
            print('Creating the neural network...', end='')

        # Define the encoders for the shared latent variables
        input1, latent1 = build_model(view1_input_dim, latent_shared_dim, network_architecture, activation, dropout, 'view1enc')
        input2, latent2 = build_model(view2_input_dim, latent_shared_dim, network_architecture, activation, dropout, 'view2enc')

        if similarity == 'cca':
            latent_loss_layer = CCALossLayer(name='cca_layer')
        elif similarity == 'mmi':
            latent_loss_layer = MMILossLayer(1.0, 1.0, 1.0, self_probs=False, name='mmi_layer')
        elif similarity == 'contrastive':
            latent_loss_layer = ContrastiveLossLayer(0.5, name='contrastive_layer')
        else:
            raise AssertionError('Unsupported embedding similarity metric: {}.'.format(similarity))
        latent11, latent22 = latent_loss_layer(latent1, latent2)

        # Define the enconders for the private latent variables
        if latent_private_dim > 0:
            _, latent1_priv = build_model(input1, latent_private_dim, network_architecture, activation, dropout, 'view1enc-priv')
            _, latent2_priv = build_model(input2, latent_private_dim, network_architecture, activation, dropout, 'view2enc-priv')
            # Regularize the private latent variables with a KL loss to a standard normal distribution
            latent11_priv = KLLossLayer(name='kl_layer1')(latent1_priv)
            latent22_priv = KLLossLayer(name='kl_layer2')(latent2_priv)
            # Concatenate the shared & private latent variables
            latent11 = keras.layers.concatenate([latent11, latent11_priv], name='bottleneck1')
            latent22 = keras.layers.concatenate([latent22, latent22_priv], name='bottleneck2')

        if cycle_loss > 0.0:
            # Define the decoders
            network_architecture.reverse()
            _, output1 = build_model(latent11, view1_input_dim, network_architecture, activation, dropout, 'view1dec')
            _, output2 = build_model(latent22, view2_input_dim, network_architecture, activation, dropout, 'view2dec')
            # Define the multiview autoencoder
            self._model = keras.Model(inputs=[input1, input2], outputs=[output1, output2], name="autoencoder")
            # Reconstruction loss between the inputs and the outputs
            reconstruction_loss = {
                "view1dec_output": "mean_squared_error",
                "view2dec_output": "mean_squared_error",
            }
            reconstruction_loss_weigths = {
                "view1dec_output": cycle_loss,
                "view2dec_output": cycle_loss,
            }
        else:
            self._model = keras.Model(inputs=[input1, input2], outputs=[latent11, latent22], name="autoencoder")
            reconstruction_loss = reconstruction_loss_weigths = None

        self._model.compile(optimizer=keras.optimizers.Adam(lr=lr, clipnorm=1.0), loss=reconstruction_loss, loss_weights=reconstruction_loss_weigths)
        # Define the encoder models for both views
        self._view1enc = keras.Model(inputs=input1, outputs=latent1, name='view1enc')
        self._view2enc = keras.Model(inputs=input2, outputs=latent2, name='view2enc')
        self._model_ready = True
        if self._verbose:
            print('Done')
            print(self._model.summary())

    def build_from_file(self, filename):
        # Restore the variables
        if self._verbose:
            print('Restoring the model...', end='')
        # Read the model
        with open("{}.json".format(filename), 'r') as json_file:
            model_info = json.load(json_file)
        self._dtw_distance = model_info['DTW_distance']
        # Restore the whole model
        self._model = keras.models.load_model(model_info['Model_filename'],
                                              custom_objects={'lrelu': lrelu,
                                                              'CCALossLayer': CCALossLayer,
                                                              'MMILossLayer': MMILossLayer,
                                                              'ContrastiveLossLayer': ContrastiveLossLayer,
                                                              'KLLossLayer': KLLossLayer})
        # Reconstruct the encoders graphs from the autoencoder model
        self._view1enc = keras.Model(inputs=self._model.get_layer('view1enc_input').input, outputs=self._model.get_layer('view1enc_output').output, name='view1enc')
        self._view2enc = keras.Model(inputs=self._model.get_layer('view2enc_input').input, outputs=self._model.get_layer('view2enc_output').output, name='view2enc')
        print('Done')
        self._model_ready = True

    def save(self, filename):
        if not self._model_ready:
            raise AssertionError('The model has not yet been initialised by calling to build() or build_from_file().')
        if self._verbose:
            print('Saving the model in {}'.format(filename))
        # Create a dictionary with the model meta-info and save it as a JSON file
        model_info = {'DTW_distance': self._dtw_distance,
                      'Model_filename': '{}.h5'.format(filename)}
        with open("{}.json".format(filename), 'w') as outfile:
            json.dump(model_info, outfile)
        # Save the model itself
        self._model.save('{}.h5'.format(filename))

    def fit(self, view1_train_dataset, view2_train_dataset, view1_val_dataset=None, view2_val_dataset=None, max_iters=100, epochs_per_dtw=1, batch_size=500, dtw_metric='euclidean', noise_std=0.0):
        if not self._model_ready:
            raise AssertionError('The model has not yet been initialised by calling to build() or build_from_file().')

        save_plots = False
        plots_dir = 'train_dtw_plots'
        if __debug__:
            save_plots = True
            if not os.path.exists(plots_dir):
                os.mkdir(plots_dir)

        self._dtw_distance = dtw_metric

        if view1_val_dataset and view2_val_dataset:
            checkpoint_file = 'dcca_weights.h5'

        if self._verbose:
            print('Starting training...')

        train_alignments = []
        best_val_error = float('Inf')
        last_best_iter = 0
        for i in range(max_iters):
            start_time = time.time()

            if not train_alignments:
                train_alignments = init_alignments(view1_train_dataset, view2_train_dataset)

            # Align both views using the existing align paths and create the datasets for Keras
            view1, view2 = align_data(view1_train_dataset, view2_train_dataset, train_alignments)
            view1_noisy = preprocess.add_noise(view1, noise_std) if noise_std > 0.0 else view1
            view2_noisy = preprocess.add_noise(view2, noise_std) if noise_std > 0.0 else view2

            # Fit the models using the realigned data.
            # self._model.fit([view1, view2], np.zeros(view1.shape[0]), batch_size=batch_size, epochs=epochs_per_dtw, shuffle=True, verbose=2)  # , callbacks=[tensorboard_callback]
            self._model.fit([view1_noisy, view2_noisy], [view1, view2], batch_size=batch_size, epochs=epochs_per_dtw, shuffle=True, verbose=2)  # , callbacks=[tensorboard_callback]

            # Predict the latent variables for both views
            view1_train_latent = predict_variables(self._view1enc, view1_train_dataset)
            view2_train_latent = predict_variables(self._view2enc, view2_train_dataset)

            # Align the predictions
            train_alignments, train_dtw_cost = compute_alignments(view1_train_latent, view2_train_latent, dtw_metric, save_plots, '{}/train_iter{}'.format(plots_dir, i))
            duration = time.time() - start_time

            if self._verbose:
                print('Epoch %4d: Training= %.3f' % (i, train_dtw_cost), end='')

            # Early stopping
            if view1_val_dataset and view2_val_dataset:
                # Predict the latent factors for the validation sequences and align them
                view1_val_latent = predict_variables(self._view1enc, view1_val_dataset)
                view2_val_latent = predict_variables(self._view2enc, view2_val_dataset)
                _, val_dtw_cost = compute_alignments(view1_val_latent, view2_val_latent, dtw_metric, save_plots, '{}/val_iter{}'.format(plots_dir, i))
                print('    Validation= %.3f' % val_dtw_cost, end='')

                if val_dtw_cost < best_val_error:
                    best_val_error = val_dtw_cost
                    last_best_iter = i
                    # Save the weights
                    self._model.save_weights(checkpoint_file)
                    if self._verbose:
                        print('   (%.1f sec)' % duration)
                else:
                    if self._verbose:
                        print('** (%.1f sec)' % duration)
                    if i - last_best_iter >= _EARLY_STOPPING_PATIENCE:
                        break
            else:
                print('   (%.1f sec)' % duration)

        if self._verbose:
            print('Training completed!!')

        # If a validation set was used, restore the best weights
        if view1_val_dataset and view2_val_dataset:
            self._model.load_weights(checkpoint_file)
            os.remove(checkpoint_file)

    def align(self, x, y):
        # Predict the latent factors both sequences
        view1_latent = predict_variables(self._view1enc, x)
        view2_latent = predict_variables(self._view2enc, y)

        # Align the latent variables using DTW
        cost_matrix, path = dtw(view1_latent.T, view2_latent.T, metric=self._dtw_distance, backtrack=True, global_constraints=True, band_rad=_DTW_BAND_RAD)
        # cost_matrix, path = dtw(view1_latent.T, view2_latent.T, metric=self._dtw_distance, backtrack=True, global_constraints=False)

        # Apply the alignments to the original sequences
        p1 = np.flip(path[:, 0], axis=0)
        p2 = np.flip(path[:, 1], axis=0)
        # return view1_seq[p1, :], view2_seq[p2, :], cost_matrix, (p1, p2)
        return p1, p2, cost_matrix

    def predict_latent_variables_view1(self, seq):
        return predict_variables(self._view1enc, seq)

    def predict_latent_variables_view2(self, seq):
        return predict_variables(self._view2enc, seq)
