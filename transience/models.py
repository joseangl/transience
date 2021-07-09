#!/usr/bin/env python3


from tensorflow import keras
from . import layers


def build_encoder(model_type, input_dim, output_dim, name='', **kwargs):
    if model_type.lower() == 'dnn':
        input_layer, output_layer = build_dnn(input_dim, output_dim, name, **kwargs)
    elif model_type.lower() == 'resnet':
        pass
    elif model_type.lower() == 'densenet':
        pass
    else:
        raise AssertionError('Unsupported mode: {}.'.format(model_type))


def build_dnn(input_dim, output_dim, network_arch, activation='relu', dropout=0.0, name=''):
    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    # reg_par = 1e-5
    act_fun = layers.get_activation(activation)

    input_layer = keras.layers.Input(shape=(input_dim,), name='{}_input'.format(name))
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
