# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.converter_ir.op_adapter import NoopOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import ConverterError


class IdentityNLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes):
            output_names = [n.name for n in nodes[0].outputs]
            super(IdentityNLayerResolver.Descriptor, self).__init__('IdentityN', name, nodes, output_names)

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['IdentityN'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)

        descriptors = []
        for match in matches:
            identity_n_op = match['root']
            descriptors.append(IdentityNLayerResolver.Descriptor(str(identity_n_op.name), match.consumed_nodes))
        return descriptors


class IdentityNLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: IdentityNLayerResolver.Descriptor
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_names = descriptor.output_names

        if len(input_names) != len(output_names):
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_IDENTITY_N_DIFF_IN_OUT')(descriptor.layer_name))

        # add no-op which will be squashed or pruned in IR optimization stage
        last_added_node = None
        for i in range(len(input_names)):
            in_name = input_names[i]
            out_name = output_names[i]
            op_name = "{}_noop{}".format(descriptor.layer_name, i)
            last_added_node = ir_graph.add(NoopOp(op_name), in_name, out_name)

        # any better value...?
        return last_added_node
