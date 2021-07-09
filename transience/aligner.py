#!/usr/bin/env python3

import time
import datetime
import os
import json
import tempfile
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd
from tensorflow import keras
import numpy as np
from librosa.sequence import dtw
import matplotlib.pyplot as plt
import librosa.display
from . import preprocess

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


class MultiviewAligner(object):
    def __init__(self, verbose=True, chk_file=None):
        self._verbose = verbose
        self._model = None
        self._dtw_distance = None
        self._model_ready = False
        self._chk_file = chk_file
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
        # if self._verbose:
        #     print('Saving the model in {}'.format(filename))
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
            checkpoint_file = tempfile.NamedTemporaryFile(delete=False)
            checkpoint_file.close()

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
                    self._model.save_weights(checkpoint_file.name)
                    if self._chk_file:
                        self.save(self._chk_file)
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
            self._model.load_weights(checkpoint_file.name)
            os.remove(checkpoint_file.name)

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
