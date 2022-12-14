# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np
from qti.aisw.converters.common.converter_ir.op_adapter import (
    ConstantOp,
    ElementwiseAndOp,
    ElementwiseDivOp,
    ElementwiseEqualOp,
    ElementwiseFloorDivOp,
    ElementwiseGreaterOp,
    ElementwiseGreaterEqualOp,
    ElementwiseLessOp,
    ElementwiseLessEqualOp,
    ElementwiseMaxOp,
    ElementwiseMinOp,
    ElementwiseNotEqualOp,
    ElementwiseOrOp,
    ElementwiseProductOp,
    ElementwiseSelectOp,
    ElementwiseSubOp,
    ElementwiseSumOp,
)
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from abc import ABC, abstractmethod
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class EltWiseBiasaddLayerResolver(LayerResolver):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, bias_name, bias):
            super(EltWiseBiasaddLayerResolver.Descriptor, self).__init__('BiasAdd', name, nodes)
            self.bias_tensor = bias
            self.bias_name = bias_name

    def __init__(self):
        super(EltWiseBiasaddLayerResolver, self).__init__()

        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['BiasAdd']),
            NonConsumableConverterSequenceNode('bias', ['?']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence.set_inputs('root', ['input', 'bias'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
            eltwise_op = match['root']
            bias_op = match['bias']
            bias = graph_helper.evaluate_tensor_output(bias_op.outputs[0])
            descriptor = EltWiseBiasaddLayerResolver.Descriptor(str(eltwise_op.name),
                                                                match.consumed_nodes,
                                                                str(bias_op.name),
                                                                bias)
            descriptors.append(descriptor)

        return descriptors


class EltWiseBiasaddLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseBiasaddResolver.Descriptor
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]

        return ir_graph.add(ElementwiseSumOp(descriptor.layer_name),
                            input_names,
                            output_name)


class EltWiseLayerResolver(LayerResolver, ABC, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, op_type, name, nodes, output_names=None, unary=False):
            super(EltWiseLayerResolver.Descriptor, self).__init__(op_type, name, nodes, output_names=output_names)
            self.unary = unary

    def __init__(self, layer_type, op_type, descriptor_class):
        super(EltWiseLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._op_type = op_type
        self._descriptor_class = descriptor_class

        self.sequence = GraphSequence([
            ConverterSequenceNode('root', self._op_type),
        ])
        self.sequence.set_outputs(['root'])

        # used for FRVSR
        self.sequence3 = GraphSequence([
            NonConsumableConverterSequenceNode('input1', ['Shape']),
            NonConsumableConverterSequenceNode('input2', ['?']),
            NonConsumableConverterSequenceNode('strided_slice_2', ['StridedSlice']),
            NonConsumableConverterSequenceNode('strided_slice_1', ['StridedSlice']),
            ConverterSequenceNode('range', ['Range']),
            ConverterSequenceNode('size', ['Size']),
            ConverterSequenceNode('size_1', ['Size']),
            ConverterSequenceNode('range1', ['Range']),
            ConverterSequenceNode('reshape', ['Reshape']),
            ConverterSequenceNode('packed', ['Pack']),
            ConverterSequenceNode('reshape1', ['Reshape']),
            ConverterSequenceNode('reshape2', ['Reshape']),
            ConverterSequenceNode('ones', ['Fill']),
            ConverterSequenceNode('reshape3', ['Reshape']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('mul1', ['Mul']),
            ConverterSequenceNode('stack', ['Pack']),
            ConverterSequenceNode('cast', ['Cast']),
            ConverterSequenceNode('const_frvsr', ['ExpandDims']),
            ConverterSequenceNode('root', self._op_type),
            NonConsumableConverterSequenceNode('stub_19', ['?']),
            NonConsumableConverterSequenceNode('stub_20', ['?']),
            NonConsumableConverterSequenceNode('stub_21', ['?']),
            NonConsumableConverterSequenceNode('stub_22', ['?']),
            NonConsumableConverterSequenceNode('stub_23', ['?']),
            NonConsumableConverterSequenceNode('stub_24', ['?']),
            ConverterSequenceNode('stub_25', ['?']),
            ConverterSequenceNode('stub_26', ['?']),
            ConverterSequenceNode('stub_27', ['?']),
            ConverterSequenceNode('stub_28', ['?']),
            ConverterSequenceNode('stub_29', ['?']),
            ConverterSequenceNode('stub_30', ['?']),
            ConverterSequenceNode('stub_31', ['?']),
            ConverterSequenceNode('stub_32', ['?']),
            ConverterSequenceNode('stub_33', ['?']),
            ConverterSequenceNode('stub_34', ['?']),
        ])
        self.sequence3.set_inputs('strided_slice_2', ['input1', 'stub_19', 'stub_20', 'stub_21'])
        self.sequence3.set_inputs('range', ['stub_25', 'strided_slice_2', 'stub_26'])
        self.sequence3.set_inputs('reshape', ['range', 'stub_29'])
        self.sequence3.set_inputs('reshape2', ['reshape', 'stub_31'])
        self.sequence3.set_inputs('size', ['range'])
        self.sequence3.set_inputs('size_1', ['range1'])
        self.sequence3.set_inputs('packed', ['size_1', 'size'])
        self.sequence3.set_inputs('ones', ['packed', 'stub_32'])
        self.sequence3.set_inputs('mul', ['reshape2', 'ones'])
        self.sequence3.set_inputs('strided_slice_1', ['input1', 'stub_22', 'stub_23', 'stub_24'])
        self.sequence3.set_inputs('range1', ['stub_27', 'strided_slice_1', 'stub_28'])
        self.sequence3.set_inputs('reshape1', ['range1', 'stub_30'])
        self.sequence3.set_inputs('reshape3', ['reshape1', 'stub_33'])
        self.sequence3.set_inputs('mul1', ['reshape3', 'ones'])
        self.sequence3.set_inputs('stack', ['mul1', 'mul'])
        self.sequence3.set_inputs('cast', ['stack'])
        self.sequence3.set_inputs('const_frvsr', ['cast', 'stub_34'])
        self.sequence3.set_inputs('root', ['const_frvsr', 'input2'])
        self.sequence3.set_outputs(['root'])

        # used for FRVSR
        self.sequence4 = GraphSequence([
            NonConsumableConverterSequenceNode('input1', ['Shape']),
            NonConsumableConverterSequenceNode('input2', ['?']),
            ConverterSequenceNode('strided_slice', ['StridedSlice']),
            ConverterSequenceNode('stub_1', ['Const']),
            ConverterSequenceNode('stub_2', ['Const']),
            ConverterSequenceNode('stub_3', ['Const']),
            ConverterSequenceNode('const_frvsr', ['Cast']),
            ConverterSequenceNode('root', self._op_type),
        ])
        self.sequence4.set_inputs('strided_slice', ['input1', 'stub_1', 'stub_2', 'stub_3'])
        self.sequence4.set_inputs('const_frvsr', ['strided_slice'])
        self.sequence4.set_inputs('root', ['const_frvsr', 'input2'])
        self.sequence4.set_outputs(['root'])

        self.sequence5 = GraphSequence([
            ConverterSequenceNode('root', self._op_type),
            NonConsumableConverterSequenceNode('const_frvsr', ['StridedSlice']),
            NonConsumableConverterSequenceNode('begin', ['Const']),
            NonConsumableConverterSequenceNode('end', ['Const']),
            NonConsumableConverterSequenceNode('strides', ['Const']),
            NonConsumableConverterSequenceNode('input', ['Shape']),
            NonConsumableConverterSequenceNode('input1', ['?']),
        ])
        self.sequence5.set_inputs('const_frvsr', ['input', 'begin', 'end', 'strides'])
        self.sequence5.set_inputs('root', ['input1', 'const_frvsr'])
        self.sequence5.set_outputs(['root'])

        # used for FRVSR
        self.sequence6 = GraphSequence([
            NonConsumableConverterSequenceNode('input1', ['Shape']),
            NonConsumableConverterSequenceNode('input2', ['?']),
            ConverterSequenceNode('strided_slice', ['StridedSlice']),
            ConverterSequenceNode('stub_1', ['Const']),
            ConverterSequenceNode('stub_2', ['Const']),
            ConverterSequenceNode('stub_3', ['Const']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('sub_const', ['Const']),
            ConverterSequenceNode('const_frvsr', ['Cast']),
            ConverterSequenceNode('root', self._op_type),
        ])
        self.sequence6.set_inputs('strided_slice', ['input1', 'stub_1', 'stub_2', 'stub_3'])
        self.sequence6.set_inputs('sub', ['sub_const', 'strided_slice'])
        self.sequence6.set_inputs('const_frvsr', ['sub'])
        self.sequence6.set_inputs('root', ['const_frvsr', 'input2'])
        self.sequence6.set_outputs(['root'])

        self.sequence7 = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['Shape']),
            NonConsumableConverterSequenceNode('input1', ['?']),
            NonConsumableConverterSequenceNode('strided_slice_1', ['StridedSlice']),
            NonConsumableConverterSequenceNode('range', ['Range']),
            NonConsumableConverterSequenceNode('strided_slice', ['StridedSlice']),
            NonConsumableConverterSequenceNode('strided_slice_2', ['StridedSlice']),
            NonConsumableConverterSequenceNode('mul2', ['Mul']),
            NonConsumableConverterSequenceNode('pack', ['Pack']),
            NonConsumableConverterSequenceNode('mul3', ['Mul']),
            ConverterSequenceNode('const_frvsr', ['Reshape']),
            ConverterSequenceNode('root', self._op_type),
            NonConsumableConverterSequenceNode('stub_10', ['?']),
            NonConsumableConverterSequenceNode('stub_11', ['?']),
            NonConsumableConverterSequenceNode('stub_12', ['?']),
            NonConsumableConverterSequenceNode('stub_13', ['?']),
            NonConsumableConverterSequenceNode('stub_14', ['?']),
            NonConsumableConverterSequenceNode('stub_15', ['?']),
            NonConsumableConverterSequenceNode('stub_16', ['?']),
            NonConsumableConverterSequenceNode('stub_17', ['?']),
            NonConsumableConverterSequenceNode('stub_18', ['?']),
            NonConsumableConverterSequenceNode('stub_19', ['?']),
            NonConsumableConverterSequenceNode('stub_20', ['?']),
            NonConsumableConverterSequenceNode('stub_21', ['?']),
        ])
        self.sequence7.set_inputs('strided_slice', ['input', 'stub_15', 'stub_16', 'stub_17'])
        self.sequence7.set_inputs('range', ['strided_slice', 'stub_13', 'stub_14'])
        self.sequence7.set_inputs('strided_slice_1', ['input', 'stub_10', 'stub_11', 'stub_12'])
        self.sequence7.set_inputs('mul2', ['range', 'strided_slice_1'])
        self.sequence7.set_inputs('strided_slice_2', ['input', 'stub_18', 'stub_19', 'stub_20'])
        self.sequence7.set_inputs('mul3', ['mul2', 'strided_slice_2'])
        self.sequence7.set_inputs('pack', ['strided_slice', 'stub_21'])
        self.sequence7.set_inputs('const_frvsr', ['mul3', 'pack'])
        self.sequence7.set_inputs('root', ['const_frvsr', 'input1'])
        self.sequence7.set_outputs(['root'])

        self.sequence_with_identity = GraphSequence([
            ConverterSequenceNode('root', self._op_type),
            ConverterSequenceNode('identity', ['Identity']),
            NonConsumableConverterSequenceNode('input1', ['?']),
            NonConsumableConverterSequenceNode('input2', ['?'])
        ])
        self.sequence_with_identity.set_inputs('identity', ['root'])
        self.sequence_with_identity.set_inputs('root', ['input1', 'input2'])
        self.sequence_with_identity.set_outputs(['identity'])

        self.sequence_with_const_input = GraphSequence([
            ConverterSequenceNode('root', self._op_type),
            NonConsumableConverterSequenceNode('const', ['Const']),
            NonConsumableConverterSequenceNode('other', ['?'])
        ])

        self.sequence_with_const_input.set_inputs('root', ['const', 'other'])
        self.sequence_with_const_input.set_outputs(['root'])

        self.sequence_with_const_or_identity_input = GraphSequence([
            ConverterSequenceNode('root', self._op_type),
            NonConsumableConverterSequenceNode('const', ['Const', 'Identity']),
            NonConsumableConverterSequenceNode('other', ['?'])
        ])
        self.sequence_with_const_or_identity_input.set_inputs('root', ['const', 'other'])
        self.sequence_with_const_or_identity_input.set_outputs(['root'])

        self.sequence_with_const_input_and_identity = GraphSequence([
            ConverterSequenceNode('root', self._op_type),
            ConverterSequenceNode('identity', ['Identity']),
            NonConsumableConverterSequenceNode('const', ['Const', 'Identity']),
            NonConsumableConverterSequenceNode('other', ['?'])
        ])
        self.sequence_with_const_input_and_identity.set_inputs('root', ['const', 'other'])
        self.sequence_with_const_input_and_identity.set_inputs('identity', ['root'])
        self.sequence_with_const_input_and_identity.set_outputs(['identity'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []

        # Matches basic sequence with root and n-inputs
        for match in graph_matcher.match_sequence(self.sequence):
            eltwise_op = match['root']
            descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), match.consumed_nodes,
                                                unary=(eltwise_op.inputs[0] == eltwise_op.inputs[1]))
            descriptors.append(descriptor)

        non_const_input_sequences = [self.sequence_with_identity]
        for sequence in non_const_input_sequences:
            for match in graph_matcher.match_sequence(sequence):
                eltwise_op = match['root']
                descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), match.consumed_nodes)
                descriptors.append(descriptor)

        frvsr_sequences = [self.sequence3, self.sequence4, self.sequence5, self.sequence6, self.sequence7]
        for sequence in frvsr_sequences:
            for match in graph_matcher.match_sequence(sequence):
                eltwise_op = match['root']
                eltwise_descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), [eltwise_op])
                descriptors.append(eltwise_descriptor)

                const_op = match['const_frvsr']
                consumed_nodes = match.consumed_nodes
                consumed_nodes.remove(eltwise_op)
                if len(consumed_nodes) == 0:
                    consumed_nodes.append(const_op)

                const_tensor = graph_helper.evaluate_tensor_output(const_op.outputs[0])

                # Sub RealDiv Mul Min Max Ops support dynamic broadcasting
                # Eltwise sum op is not broadcasted here. It is checked for squashing in optimizations step
                # If squash not possible the OP is broadcasted in OP graph optimizations
                eltwise_shape = graph_helper.get_op_output_shape(const_op)
                if not eltwise_shape:
                    eltwise_shape = [1]

                const_descriptor = ConstantLayerResolver.Descriptor(str(const_op.name), consumed_nodes,
                                                                    const_tensor, eltwise_shape, eltwise_descriptor)
                descriptors.append(const_descriptor)

        const_input_sequences = [self.sequence_with_const_input_and_identity, self.sequence_with_const_input, self.sequence_with_const_or_identity_input]
        for sequence in const_input_sequences:
            for match in graph_matcher.match_sequence(sequence):
                eltwise_op = match['root']
                eltwise_descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name),
                                                            match.consumed_nodes)
                descriptors.append(eltwise_descriptor)

                const_op = match['const']
                const_consumed_ops = [const_op]
                while const_op.type == 'Identity':
                    const_op = const_op.inputs[0].op
                    const_consumed_ops.append(const_op)

                if const_op.type != 'Const':
                    continue

                const_tensor = graph_helper.evaluate_tensor_output(const_op.outputs[0])

                # Sub RealDiv Mul Min Max Ops support dynamic broadcasting
                # Eltwise sum op is not broadcasted here. It is checked for squashing in optimizations step
                # If squash not possible the OP is broadcasted in OP graph optimizations
                for op_type in self._op_type:
                    eltwise_shape = graph_helper.get_op_output_shape(const_op)
                    if not eltwise_shape:
                        eltwise_shape = [1]

                const_descriptor = ConstantLayerResolver.Descriptor(str(const_op.name), const_consumed_ops,
                                                                    const_tensor, eltwise_shape, eltwise_descriptor)
                descriptors.append(const_descriptor)

        return descriptors


class EltWiseLayerBuilder(LayerBuilder, ABC):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseAndLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]

        # In case there is one input used twice, duplicate the input name
        if descriptor.unary:
            input_names = input_names * 2

        return ir_graph.add(self._op_class(descriptor.layer_name),
                            input_names,
                            output_name)


class EltWiseAndLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseAndLayerResolver, self).__init__('ElementWiseAnd', ['LogicalAnd'],
                                                      EltWiseAndLayerResolver.Descriptor)


class EltWiseAndLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseAndLayerBuilder, self).__init__()
        self._op_class = ElementwiseAndOp


class EltWiseEqualLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseEqualLayerResolver, self).__init__('ElementWiseEqual', ['Equal'],
                                                        EltWiseEqualLayerResolver.Descriptor)


class EltWiseFloorDivLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseFloorDivLayerBuilder, self).__init__()
        self._op_class = ElementwiseFloorDivOp


class EltWiseFloorDivLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseFloorDivLayerResolver, self).__init__('ElementWiseFloorDiv', ['FloorDiv'],
                                                           EltWiseFloorDivLayerResolver.Descriptor)


class EltWiseEqualLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseEqualLayerBuilder, self).__init__()
        self._op_class = ElementwiseEqualOp


class EltWiseGreaterLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseGreaterLayerResolver, self).__init__('ElementWiseGreater', ['Greater'],
                                                          EltWiseGreaterLayerResolver.Descriptor)


class EltWiseGreaterLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseGreaterLayerBuilder, self).__init__()
        self._op_class = ElementwiseGreaterOp


class EltWiseGreaterEqualLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseGreaterEqualLayerResolver, self).__init__('ElementWiseGreaterEqual', ['GreaterEqual'],
                                                               EltWiseGreaterEqualLayerResolver.Descriptor)


class EltWiseGreaterEqualLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseGreaterEqualLayerBuilder, self).__init__()
        self._op_class = ElementwiseGreaterEqualOp


class EltWiseLessLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseLessLayerResolver, self).__init__('ElementWiseLess', ['Less'], EltWiseLessLayerResolver.Descriptor)


class EltWiseLessLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseLessLayerBuilder, self).__init__()
        self._op_class = ElementwiseLessOp


class EltWiseLessEqualLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseLessEqualLayerResolver, self).__init__('ElementWiseLessEqual', ['LessEqual'],
                                                            EltWiseLessEqualLayerResolver.Descriptor)


class EltWiseLessEqualLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseLessEqualLayerBuilder, self).__init__()
        self._op_class = ElementwiseLessEqualOp


class EltWiseNotEqualLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseNotEqualLayerResolver, self).__init__('ElementWiseNotEqual', ['NotEqual'],
                                                           EltWiseNotEqualLayerResolver.Descriptor)


class EltWiseNotEqualLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseNotEqualLayerBuilder, self).__init__()
        self._op_class = ElementwiseNotEqualOp


class EltWiseOrLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseOrLayerResolver, self).__init__('ElementWiseOr', ['LogicalOr'],
                                                     EltWiseOrLayerResolver.Descriptor)


class EltWiseOrLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseOrLayerBuilder, self).__init__()
        self._op_class = ElementwiseOrOp


class EltWiseSelectLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseSelectLayerResolver, self).__init__('ElementWiseSelect', ['Select'],
                                                         EltWiseSelectLayerResolver.Descriptor)


class EltWiseSelectLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseSelectLayerBuilder, self).__init__()
        self._op_class = ElementwiseSelectOp


class EltWiseSumLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseSumLayerResolver, self).__init__('ElementWiseSum', ['Add', 'AddV2'],
                                                      EltWiseSumLayerResolver.Descriptor)


class EltWiseSumLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseSumLayerBuilder, self).__init__()
        self._op_class = ElementwiseSumOp


class EltWiseSubLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseSubLayerResolver, self).__init__('ElementWiseSub', ['Sub'], EltWiseSubLayerResolver.Descriptor)


class EltWiseSubLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseSubLayerBuilder, self).__init__()
        self._op_class = ElementwiseSubOp


class EltWiseMulLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseMulLayerResolver, self).__init__('ElementWiseMul', ['Mul'], EltWiseMulLayerResolver.Descriptor)


class EltWiseMulLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseMulLayerBuilder, self).__init__()
        self._op_class = ElementwiseProductOp


class EltWiseMaxLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseMaxLayerResolver, self).__init__('ElementWiseMax', ['Maximum'], EltWiseMaxLayerResolver.Descriptor)


class EltWiseMaxLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseMaxLayerBuilder, self).__init__()
        self._op_class = ElementwiseMaxOp


class EltWiseMinLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseMinLayerResolver, self).__init__('ElementWiseMin', ['Minimum'], EltWiseMinLayerResolver.Descriptor)


class EltWiseMinLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseMinLayerBuilder, self).__init__()
        self._op_class = ElementwiseMinOp


class EltWiseDivLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseDivLayerResolver, self).__init__('ElementWiseDiv', ['RealDiv'], EltWiseDivLayerResolver.Descriptor)


class EltWiseDivLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseDivLayerBuilder, self).__init__()
        self._op_class = ElementwiseDivOp

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):

        constant_input_descriptor = [d for d in input_descriptors if isinstance(d, ConstantLayerResolver.Descriptor)]
        if len(constant_input_descriptor) == 1 and np.all(constant_input_descriptor[0].value == 1):
            descriptor.set_ignored(True)
            constant_input_descriptor[0].set_ignored(True)
