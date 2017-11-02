import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from luminoth.models.base import FullyConvolutionalNetwork
from luminoth.models.ssd.ssd_proposal import SSDProposal
from luminoth.models.ssd.ssd_target import SSDTarget
from luminoth.models.ssd.ssd_utils import generate_anchors_reference
from luminoth.utils.config import get_base_config
from luminoth.utils.vars import variable_summaries, get_saver


DEFAULT_ENDPOINTS = {
    'ssd_1': (10, 10),
    'ssd_2': (5, 5),
    'ssd_3': (3, 3),
    'ssd_4': (1, 1),
}


class SSD(snt.AbstractModule):
    """TODO
    """

    base_config = get_base_config(__file__)

    def __init__(self, config, name='ssd'):
        super(SSD, self).__init__(name=name)

        # Main configuration object, it holds not only the necessary
        # information for this module but also configuration for each of the
        # different submodules.
        self._config = config

        # Total number of classes to classify.
        self._num_classes = config.model.network.num_classes

        # Turn on debug mode with returns more Tensors which can be used for
        # better visualization and (of course) debugging.
        self._debug = config.train.debug
        self._seed = config.train.seed

        # Anchor config, check out the docs of base_config.yml for a better
        # understanding of how anchors work.
        self._anchor_max_scale = config.model.anchors.max_scale
        self._anchor_min_scale = config.model.anchors.min_scale
        self._anchor_ratios = np.array(config.model.anchors.ratios)

        # Name of endpoints from the base network, the fully convolutional
        # base network and the '4' default endpoints of ssd architecture
        self._endpoints = (config.model.base_network.endpoints +
                           config.model.base_network.fc_endpoints +
                           ['ssd_1'] + ['ssd_2'] + ['ssd_3'] + ['ssd_4'])

        # Outputs of the endpoints
        self._endpoints_outputs = (
            config.model.base_network.endpoints_output +
            config.model.base_network.fc_endpoints_output +
            [DEFAULT_ENDPOINTS['ssd_1']] + [DEFAULT_ENDPOINTS['ssd_2']] +
            [DEFAULT_ENDPOINTS['ssd_3']] + [DEFAULT_ENDPOINTS['ssd_4']]
        )

        # Total number of anchors per point, per endpoint.
        self._anchors_per_point = config.model.anchors.anchors_per_point

        # Calculate the anchors for each endpoint (feature map)
        self.anchors = self.generate_anchors(
            self._anchors_per_point, self._endpoints, self._endpoints_outputs,
            self._anchor_min_scale, self._anchor_max_scale, self._anchor_ratios
        )

        # Weight for the localization loss
        self._loc_loss_weight = config.model.loss.localization_loss_weight
        self._losses_collections = ['ssd_losses']

        # We want the pretrained model to be outside the ssd name scope.
        self.base_network = FullyConvolutionalNetwork(
            config.model.base_network, parent_name=self.module_name
        )

    def _build(self, image, gt_boxes=None, is_training=True):
        """
        TODO
        Returns bounding boxes and classification probabilities.

        Args:
            image: A tensor with the image.
                Its shape should be `(1, height, width, 3)`.
            gt_boxes: A tensor with all the ground truth boxes of that image.
                Its shape should be `(num_gt_boxes, 5)`
                Where for each gt box we have (x1, y1, x2, y2, label),
                in that order.
            is_training: A boolean to whether or not it is used for training.

        Returns:

        """
        if gt_boxes is not None:
            gt_boxes = tf.cast(gt_boxes, tf.float32)

        # Dictionary with the endpoints from the base network
        base_network_endpoints = self.base_network(
            image, is_training=is_training)

        net = base_network_endpoints[
            self._config.model.base_network.hook_endpoint]

        with tf.variable_scope('ssd_1'):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
            net = tf.pad(net, paddings, mode='CONSTANT')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3',
                              padding='VALID')
            base_network_endpoints['ssd_1'] = net
        with tf.variable_scope('ssd_2'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
            net = tf.pad(net, paddings, mode='CONSTANT')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3',
                              padding='VALID')
            base_network_endpoints['ssd_2'] = net
        with tf.variable_scope('ssd_3'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3',
                              padding='VALID')
            base_network_endpoints['ssd_3'] = net
        with tf.variable_scope('ssd_4'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3',
                              padding='VALID')
            base_network_endpoints['ssd_4'] = net

        # Do the predictions for each feature map
        predictions = {}
        for ind, endpoint in enumerate(self._endpoints):
            inputs = base_network_endpoints[endpoint]

            num_anchors = self._anchors_per_point[ind]
            # Location predictions
            num_loc_pred = num_anchors * 4
            loc_pred = slim.conv2d(inputs, num_loc_pred, [3, 3],
                                   scope=endpoint + '/conv_loc',
                                   padding='SAME')
            loc_pred = tf.reshape(loc_pred,
                                  [self._endpoints_outputs[ind][0],
                                   self._endpoints_outputs[ind][1],
                                   num_anchors, 4])

            # Class predictions
            num_cls_pred = num_anchors * (self._num_classes + 1)
            cls_pred = slim.conv2d(inputs, num_cls_pred, [3, 3],
                                   scope=endpoint + '/conv_cls',
                                   padding='SAME')
            cls_pred = tf.reshape(cls_pred,
                                  [self._endpoints_outputs[ind][0],
                                   self._endpoints_outputs[ind][1],
                                   num_anchors, (self._num_classes + 1)])

            predictions[endpoint] = {}
            if self._debug:
                predictions[endpoint]['loc_pred'] = loc_pred
                predictions[endpoint]['cls_pred'] = cls_pred
                predictions[endpoint]['prob'] = slim.softmax(cls_pred)

        # Get all_anchors from all the endpoints
        pass
        # Get the proposals and save the result
        pass
        # Get the targets and returns it
        pass

        # TODO add variable summaries

        result = {'predictions': predictions}

        if self._debug:
            result['anchors'] = self.anchors

        return result

    def generate_anchors(self, anchors_per_point, endpoints, endpoints_outputs,
                         min_scale, max_scale, ratios):
        """
        Returns a dictionary containing the anchors per endpoint.

        Args:
        anchors_per_point (list of int): The number of anchors that will be
            fixed in each endpoint location.
        endpoints (list of strings): The names of the endpoints, these will be
             used as keys in the dictionary returned.
        endpoints_outputs (list of (height, width)): The endpoints' otuputs.

        Returns:
        anchors: A dictionary with `endpoints` as keys and an array of anchors
            as values ('[[x_min, y_min, x_max, y_max], ...]') with shape
            (anchors_per_point[i] * endpoints_outputs[i][0]
             * endpoints_outputs[i][1], 4)
        """
        # Calculate the scales (usign scale min/max and number of endpoints).
        num_endpoints = len(endpoints)
        scales = np.zeros([num_endpoints])
        for endpoint in range(num_endpoints):
            scales[endpoint] = (
                min_scale +
                (max_scale - min_scale) * (endpoint) / (num_endpoints - 1)
            )

        # For each endpoint calculate the anchors with the appropiate size.
        anchors = {}
        for ind, endpoint in enumerate(endpoints):
            # Get the anchors reference for this endpoint
            anchor_reference = generate_anchors_reference(
                ratios, scales[ind: ind + 2], anchors_per_point[ind],
                endpoints_outputs[ind]
            )

            anchors[endpoint] = self._generate_anchors(endpoints_outputs[ind],
                                                       anchor_reference)

        return anchors

    @property
    def summary(self):
        """
        Generate merged summary of all the sub-summaries used inside the
        ssd network.
        """
        summaries = [
            tf.summary.merge_all(),
        ]

        summaries.append(
            tf.summary.merge_all(key=self._losses_collections[0])
        )

        return tf.summary.merge(summaries)

    def get_trainable_vars(self):
        """Get trainable vars included in the module.
        """
        trainable_vars = snt.get_variables_in_module(self)
        if self._config.model.base_network.trainable:
            pretrained_trainable_vars = self.base_network.get_trainable_vars()
            tf.logging.info('Training {} vars from pretrained module.'.format(
                len(pretrained_trainable_vars)))
            trainable_vars += pretrained_trainable_vars
        else:
            tf.logging.info('Not training variables from pretrained module')

        return trainable_vars

    def get_saver(self, ignore_scope=None):
        """Get an instance of tf.train.Saver for all modules and submodules.
        """
        return get_saver((self, self.base_network), ignore_scope=ignore_scope)

    def load_pretrained_weights(self):
        """Get operation to load pretrained weights from file.
        """
        return self.base_network.load_weights()
