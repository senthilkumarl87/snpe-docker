# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import DeconvolutionOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.layers.convolution import ConvolutionLayerBuilder
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.util import OperationNotFoundError


class DeConvolutionOptimizedLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, deconv_op, bias_op, weights, strides, padding_size_strategy, biases,
                     input_tensor, output_names=None):
            super(DeConvolutionOptimizedLayerResolver.Descriptor, self).__init__('Deconvolution', name, nodes,
                                                                                 output_names=output_names)
            self.deconv_op = deconv_op
            self.bias_op = bias_op
            self.weights = weights
            self.strides = strides
            self.padding_size_strategy = padding_size_strategy
            self.biases = biases
            self.input_tensor = input_tensor

        def is_input_op(self, op):
            if op == self.bias_op:
                return False
            return super(DeConvolutionOptimizedLayerResolver.Descriptor, self).is_input_op(op)

        def is_input_tensor(self, op, tensor):
            # we want to preserve FakeQuant to extract quant params. It will be removed later in transform/optimizations
            if GraphHelper.get_none_identity_input(tensor)[0].op.type == "FakeQuantWithMinMaxVars":
                return True
            if op == self.deconv_op and \
                    (tensor != self.deconv_op.inputs[2] and tensor.op.type != "FakeQuantWithMinMaxVars"):
                return False
            if op == self.deconv_op:
                return tensor == self.input_tensor

            # handle cases when fakequant and identity are ignored by determining that the fakequant or identity
            # op's input tensors were off the expected third input path
            if self.deconv_op.inputs[2].op != op:
                if not len(self.deconv_op.inputs[2].op.inputs) or not len(op.inputs):
                    return False
                expected_act_input = GraphHelper.get_stripped_input(self.deconv_op.inputs[2].op.inputs[0],
                                                                    ["FakeQuantWithMinMaxVars", "Identity"])
                actual_act_input = GraphHelper.get_stripped_input(op.inputs[0],
                                                                  ["FakeQuantWithMinMaxVars", "Identity"])
                if expected_act_input != actual_act_input:
                    return False

            return tensor == op.inputs[0]

        @property
        def output_names(self):
            if self.bias_op:
                output_name = str(self.bias_op.outputs[0].name)
            else:
                output_name = str(self.deconv_op.outputs[0].name)
            return [output_name]

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Conv2DBackpropInput'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_trans_op = match['root']
            _, weights_tensor, input_tensor = GraphHelper.get_op_input_tensors(conv_trans_op, ('?', '?', '?'))
            if weights_tensor.op.type not in ['Identity', 'Const', 'Split', 'FakeQuantWithMinMaxVars'] and \
                    not graph_helper.check_tensor_const_orgin(weights_tensor)[0]:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_DECONV_CANT_FIND_WEIGHTS_NODE'))
            strides = conv_trans_op.get_attr('strides')
            padding_size_strategy = conv_trans_op.get_attr('padding')
            weights = graph_helper.evaluate_tensor_output(weights_tensor)
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in self.sequence.output_nodes]
            bias_op = None
            try:
                output_ops = graph_helper.get_op_outputs(conv_trans_op)
                bias_op = GraphHelper.filter_single_op_by_type(output_ops, 'BiasAdd')

                _, biases = GraphHelper.get_op_input_tensors(bias_op, ('?', '?'))
                if biases.op.type not in ['Const', 'Identity']:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_DECONV_CANT_FIND_BIAS_NODE'))
                biases = graph_helper.evaluate_tensor_output(biases)
                consumed_nodes.append(bias_op)
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
            except OperationNotFoundError:
                biases = np.zeros(np.shape(weights)[-2], dtype=np.float32)

            descriptors.append(
                DeConvolutionOptimizedLayerResolver.Descriptor(str(conv_trans_op.name),
                                                               consumed_nodes,
                                                               conv_trans_op,
                                                               bias_op,
                                                               weights,
                                                               strides,
                                                               padding_size_strategy,
                                                               biases,
                                                               input_tensor,
                                                               output_names=output_op_nodes_names))
        return descriptors


class DeConvolutionLayerBuilder(LayerBuilder, object):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: DeConvolutionLayerResolver.Descriptor
        :rtype: int
        """
        input_dims = converter_context.graph_helper.get_op_output_shape(descriptor.input_tensor.op)
        bias_op = descriptor.bias_op
        if bias_op:
            output_dims = converter_context.graph_helper.get_op_output_shape(bias_op)
        else:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.deconv_op)

        pads, ir_padding_strategy = ConvolutionLayerBuilder.calculate_padding_size(input_size=output_dims[-3:-1],
                                                                                   output_size=input_dims[-3:-1],
                                                                                   strides=descriptor.strides[1:3],
                                                                                   padding_size_strategy=descriptor.padding_size_strategy,
                                                                                   filter_dims=descriptor.weights.shape,
                                                                                   dilation=[1, 1])

        weights = np.transpose(descriptor.weights, (0, 1, 3, 2)).copy()

        input_names = self.get_input_name(converter_context, descriptor, input_descriptors)

        return ir_graph.add(DeconvolutionOp(name=descriptor.layer_name,
                                            weights=weights,
                                            bias=descriptor.biases,
                                            bias_op_name=bias_op.name if bias_op else None,
                                            stridex=descriptor.strides[1],
                                            stridey=descriptor.strides[2],
                                            padding_size_strategy=ir_padding_strategy,
                                            pady_before=pads[0][0],
                                            pady_after=pads[0][1],
                                            padx_before=pads[1][0],
                                            padx_after=pads[1][1],
                                            output_width=output_dims[-2],
                                            output_height=output_dims[-3],
                                            groups=1),
                            input_names,
                            descriptor.output_names[0])
