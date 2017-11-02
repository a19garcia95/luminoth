import sonnet as snt
import tensorflow as tf
import functools

from tensorflow.contrib.slim.nets import (
    vgg,
)
import tensorflow.contrib.slim as slim

from luminoth.models.base import BaseNetwork
from luminoth.utils.checkpoint_downloader import get_checkpoint_file

VALID_ARCHITECTURES = set([
    'vgg_16',
])


class FullyConvolutionalNetwork(BaseNetwork):
    def __init__(self, config, parent_name=None, name='fc_network',
                 **kwargs):
        super(FullyConvolutionalNetwork, self).__init__(config, name=name,
                                                        **kwargs)
        if config.get('architecture') not in VALID_ARCHITECTURES:
            raise ValueError('Invalid architecture "{}"'.format(
                config.get('architecture')
            ))

        self._architecture = config.get('architecture')
        self._config = config
        self.parent_name = parent_name

    def network(self, inputs, is_training=True):
        endpoints = {}
        if self.vgg_type:
            vgg_net = functools.partial(getattr(vgg, self._architecture))
            _, vgg_endpoints = vgg_net(
                inputs,
                is_training=is_training,
                spatial_squeeze=self._config.get('spatial_squeeze', False)
            )
            vgg_endpoints = dict(vgg_endpoints)

            scope = self.module_name
            if self.parent_name:
                scope = self.parent_name + '/' + scope

            # Add padding to divide by 2 without loss
            conv4_3 = vgg_endpoints[scope + '/vgg_16/conv4/conv4_3']
            paddings = [[0, 0], [1, 0], [1, 0], [0, 0]]
            conv4_3 = tf.pad(conv4_3, paddings, mode='CONSTANT')
            endpoints['vgg_16/conv4/conv4_3'] = conv4_3

            # Add padding to divide by 2 without loss
            conv5_3 = vgg_endpoints[scope + '/vgg_16/conv5/conv5_3']
            paddings = [[0, 0], [1, 0], [1, 0], [0, 0]]
            conv5_3 = tf.pad(conv5_3, paddings, mode='CONSTANT')
            net = slim.max_pool2d(conv5_3, [3, 3], stride=1, scope='pool5',
                                  padding='SAME')

            # Additional SSD blocks.
            # Block 6: Use atrous convolution
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6',
                              padding='SAME')
            endpoints['vgg_16/fc6'] = net
            # Block 7: 1x1 conv
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            endpoints['vgg_16/fc7'] = net
            return endpoints

    def _build(self, inputs, is_training=True):
        inputs = self.preprocess(inputs)
        with slim.arg_scope(self.arg_scope):
            end_points = self.network(inputs, is_training=is_training)

            return end_points

    def load_weights(self):
        """
        Creates operations to load weigths from checkpoint for each of the
        variables defined in the module. It is assumed that all variables
        of the module are included in the checkpoint but with a different
        prefix.

        Returns:
            load_op: Load weights operation or no_op.
        """
        if self.vgg_type:
            if self._config.get('weights') is None and \
               not self._config.get('download'):
                return tf.no_op(name='not_loading_network')

            if self._config.get('weights') is None:
                # Download the weights (or used cached) if is is not specified
                # in config file.
                # Weights are downloaded by default on the ~/.luminoth folder.
                self._config['weights'] = get_checkpoint_file(
                    self._architecture)

            module_variables = snt.get_variables_in_module(
                self, tf.GraphKeys.MODEL_VARIABLES
            )
            assert len(module_variables) > 0

            scope = self.module_name
            if self.parent_name:
                scope = self.parent_name + '/' + scope
            var_to_modify = [scope + '/conv6/weights',
                             scope + '/conv6/biases',
                             scope + '/conv7/weights',
                             scope + '/conv7/biases']
            load_variables = []
            variables = (
                [(v, v.op.name) for v in module_variables if v.op.name not in
                 var_to_modify]
             )

            variable_scope_len = len(self.variable_scope.name) + 1

            for var, var_name in variables:
                checkpoint_var_name = var_name[variable_scope_len:]
                var_value = tf.contrib.framework.load_variable(
                    self._config['weights'], checkpoint_var_name
                )
                load_variables.append(
                    tf.assign(var, var_value)
                )
            with tf.variable_scope('', reuse=True):
                # Original weigths and biases
                fc6_weights = tf.get_variable(scope +
                                              '/vgg_16/fc6/weights')
                fc6_biases = tf.get_variable(scope +
                                             '/vgg_16/fc6/biases')
                fc7_weights = tf.get_variable(scope +
                                              '/vgg_16/fc7/weights')
                fc7_biases = tf.get_variable(scope +
                                             '/vgg_16/fc7/biases')

                # Weights and biases to surgery
                block6_weights = tf.get_variable(scope +
                                                 '/conv6/weights')
                block6_biases = tf.get_variable(scope +
                                                '/conv6/biases')
                block7_weights = tf.get_variable(scope +
                                                 '/conv7/weights')
                block7_biases = tf.get_variable(scope +
                                                '/conv7/biases')

                # surgery
                load_variables.append(
                    tf.assign(block6_weights, fc6_weights[::3, ::3, :, ::4]))
                load_variables.append(
                    tf.assign(block6_biases, fc6_biases[::4]))
                load_variables.append(
                    tf.assign(block7_weights, fc7_weights[:, :, ::4, ::4]))
                load_variables.append(
                    tf.assign(block7_biases, fc7_biases[::4]))

            tf.logging.info(
                'Constructing op to load {} variables from pretrained '
                'checkpoint {}'.format(
                    len(load_variables), self._config['weights']
                ))

            load_op = tf.group(*load_variables)
            return load_op
