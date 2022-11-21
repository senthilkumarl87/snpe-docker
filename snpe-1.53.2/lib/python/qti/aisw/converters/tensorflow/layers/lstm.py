# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from qti.aisw.converters.common.converter_ir.op_adapter import LstmOp, ReshapeOp
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.tensorflow.layers.slice import SliceLayerResolver
from qti.aisw.converters.tensorflow.layers.pack import UnPackLayerResolver, PackLayerResolver
from qti.aisw.converters.tensorflow.layers.concat import ConcatLayerResolver
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.util import ConverterError, OperationNotFoundError, TensorNotFoundError
from qti.aisw.converters.tensorflow.sequences.lstm import cell_sequence


class LstmLayerResolver(LayerResolver, object):
    class UnrolledTimeStepDescriptor(LayerDescriptor):
        def __init__(self, name, operations, cell_input_concat_op, gates_matmul_op, gates_biases_op, cell_output_op,
                     init_cell_state_op,
                     final_cell_state_output_op,
                     forget_bias_value):
            super(LstmLayerResolver.UnrolledTimeStepDescriptor, self).__init__('LSTM', name, operations)
            self.cell_input_concat_op = cell_input_concat_op
            self.gates_matmul_op = gates_matmul_op
            self.gates_biases_op = gates_biases_op
            self.cell_output_op = cell_output_op
            self.init_cell_state_op = init_cell_state_op
            self.final_cell_state_output_op = final_cell_state_output_op
            self.forget_bias_value = forget_bias_value
            self.cell_0 = self
            self.unrolled_cells = [self]
            self._is_stacked_cell = False

        def is_input_op(self, op):
            return op == self.cell_input_concat_op or op == self.init_cell_state_op

        def is_input_tensor(self, op, tensor):
            # Ignores a static axis input which has already been consumed by the resolver
            if tensor.op.type == "Const" and (len(self.cell_input_concat_op.inputs) == 3 and
                                              tensor == self.cell_input_concat_op.inputs[-1]):
                return False
            return True

        def is_unrolled_cell_of(self, lstm_descriptor):
            """
            An LSTM is said to be an unrolled cell if each time-step is spun into its own set of (similar) ops.
            This function checks if the passed descriptor is a time-step of the current LSTM descriptor.
            This is the case when the matmul ops have the same set of weights between both descriptors.
            :param lstm_descriptor: The candidate descriptor that could be a time-step
            :return: True if weight tensors are identical, False otherwise
            """
            return self.gates_matmul_op.inputs[1].op == lstm_descriptor.gates_matmul_op.inputs[1].op

        def is_cell_of_time_step_0(self):
            """
            Checks to see if the current descriptor is the first in a series of time-steps or an LSTM descriptor
            with a single time-step.
            This is relevant when len(self.unrolled_cells) > 1
            """
            return self.cell_0 == self

        def time_steps(self):
            return len(self.unrolled_cells)

        def is_output_op(self, op):
            return op.outputs[0].name in self._output_tensor_names

        @property
        def output_names(self):
            if not self._is_stacked_cell:
                return self._output_tensor_names
            else:
                # stacked cells do not return final and hidden cell states
                return [self.rolled_cell_output_name]

        def set_is_stacked_cell(self, is_stacked_cell):
            # TO-DO: Need to evaluate if we really need this
            self._is_stacked_cell = is_stacked_cell

        @property
        def rolled_cell_output_name(self):
            """
            The output for an LSTM can be a full sequence of intermediate hidden states for all time-steps.
            This could be present in a model as a concat op whose inputs are the hidden states.
            If the concat op is found, its name is used as a rolled cell output name, otherwise,
            a tensor name is created to capture the output produced by the runtimes.
            :return:
            """
            cell_child_op = self.cell_0.child_ops[-1]
            # if intermediate outputs were already concatenated, then use existing concat output name otherwise return new name
            # Only preserve output names if it will be used by subsequent ops as tests expect name to have _all_time_steps prefix
            # TO-D0: remove cell_output restriction once test issue is resolved
            out_tensor_names = self._output_tensor_names
            if cell_child_op.type in ["Concat", "ConcatV2"] and \
                    cell_child_op.inputs[0].shape == self.cell_output_op.outputs[0].shape and len(cell_child_op.outputs[0].consumers()):
                return cell_child_op.outputs[0].name
            return '{}_all_time_steps'.format(out_tensor_names[-1])

        def get_output_names_for(self, input_tensors):
            """
            Returns the output names for a given set of input tensors. If the op is a stacked cell i.e all the hidden state outputs
            are concatenated, then the unrolled cell name is returned. Otherwise, the base class function is called.
            :type input_tensors: [tensorflow.Tensor]
            :rtype list: List of output names if any are found
            """
            if not self._is_stacked_cell:
                return super(LstmLayerResolver.UnrolledTimeStepDescriptor, self).get_output_names_for(input_tensors)
            else:
                return [t.name for t in input_tensors if t.name == self.rolled_cell_output_name]

        @property
        def _output_tensor_names(self):
            # Return a pair of final cell state and hidden state
            return [str(self.unrolled_cells[-1].final_cell_state_output_op.outputs[0].name),
                    str(self.unrolled_cells[-1].cell_output_op.outputs[0].name)]

        def returns_state(self):
            """
            Checks if any of the final states are to be consumed by non-LSTM child ops
            :return: True if consumers are found, false otherwise
            """
            last_cell = self.cell_0.unrolled_cells[-1]
            cell_state_consumers = last_cell.final_cell_state_output_op.outputs[0].consumers()
            hidden_state_consumers = last_cell.cell_output_op.outputs[0].consumers()

            return not all(consumer in self.cell_0.child_ops for consumer in hidden_state_consumers) or \
                   not all(consumer in self.cell_0.child_ops for consumer in cell_state_consumers)

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(cell_sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            """
            The LSTM op is matched as a sequence of ops which perform the following set of computations.
            The results h_t and c_t are considered to be the outputs of the descriptor that will be matched.
            Please adjust the computations to reflect any additions to the existing supported sequence.
            -  gate_data = X_t*W_input + H_t-1*W_recurrent + B
            - i_t  = sigmoid(split(gate_data)[0])
            - f_t  = sigmoid(split(gate_data)[2]) (argument +1 if unit forget bias is set)
            - g_t  = tanh(split(gate_data)[1])
            - c_t = f_t (.) c_t-1 + i_t (.) c_t
            - o_t = sigmoid(split(gate_data)[3])) if peepholes are set
            - h_t = ot (.) h(Ct)
            """
            # The cell input concat op consumes the input data [X] and initial hidden state [h_t-1] as inputs
            cell_input_concat_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/concat']

            # The gates matmul op consumes concatenated [X, h] and gate weights [ W_input, W_recurrent] as inputs
            gates_matmul_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/MatMul']

            # The gates biases op consumes X*W_input + H_t-1*W_recurrent and the gate biases [B_input, B_recurrent]
            # Following the bias add, the output is split and fed into each respective gate
            gates_biases_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd']

            # The init_cell state op consumes initial cell state [c_t-1] and output of ft to produce ft*c_t-1
            init_cell_state_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul']

            # The cell state output op is c_t above
            final_cell_state_output_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1']

            # get forget bias value
            forget_bias_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add']
            _, forget_bias_tensor = graph_helper.get_op_input_tensors(forget_bias_op, ('?', 'Const'))
            forget_bias_value = float(graph_helper.evaluate_tensor_output(forget_bias_tensor))

            # The cell output op produces the output of h_t.
            # Note that each individual time-step will be matched, and that the output of this op could be fed into
            # the cell_input_concat_op for another match (as the initial hidden state)
            cell_output_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2']

            d = LstmLayerResolver.UnrolledTimeStepDescriptor(str(cell_output_op.name),
                                                             match.consumed_nodes,
                                                             cell_input_concat_op=cell_input_concat_op,
                                                             gates_matmul_op=gates_matmul_op,
                                                             gates_biases_op=gates_biases_op,
                                                             cell_output_op=cell_output_op,
                                                             init_cell_state_op=init_cell_state_op,
                                                             final_cell_state_output_op=final_cell_state_output_op,
                                                             forget_bias_value=forget_bias_value)
            descriptors.append(d)

        if len(descriptors) == 0:
            return []

        return descriptors


class LstmLayerBuilder(LayerBuilder):
    _TENSORFLOW_INPUT_GATE_INDEX = 0
    _TENSORFLOW_FORGET_GATE_INDEX = 2
    _TENSORFLOW_OUTPUT_GATE_INDEX = 3
    _TENSORFLOW_STATE_GATE_INDEX = 1

    @classmethod
    def _add_reshape_to_restore_time_dimension(cls, ir_graph, descriptor, input_name, input_shape):
        """
        This functions inserts a reshape op from 2D to 3D before the LSTM op. This is based on an observation
        that LSTM models may have a split op to unpack the data for each individual time-step. Since we combining
        all time-steps into a single op and removing the split, we need to re-insert a reshape after the input data.
        """

        if len(input_shape) != 2:
            raise ValueError("Input shape to restore for LSTM layer: {} must be of size 2. "
                             "Got {} instead".format(descriptor.layer_name, len(input_shape)))

        reshape_layer_name = '{}_reshape'.format(descriptor.layer_name)
        reshape_output = [input_shape[0], descriptor.time_steps(), input_shape[1]]
        ir_graph.add(ReshapeOp(reshape_layer_name,
                               reshape_output),
                     input_name,
                     reshape_layer_name)
        return reshape_layer_name

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: LstmLayerResolver.UnrolledTimeStepDescriptor
        :rtype: int
        """
        input_shape = converter_context.graph_helper.get_op_output_shape(descriptor.cell_input_concat_op.inputs[0].op)
        state_shape = converter_context.graph_helper.get_op_output_shape(descriptor.cell_input_concat_op.inputs[1].op)

        # Weights are organized in I, C, F, O but converter supports I, O, F, C
        gates_weights, input_weights = self._resolve_weights(descriptor, converter_context.graph_helper, state_shape)
        gates_biases = self._resolve_biases(descriptor, converter_context.graph_helper)

        def is_cell_input_descriptor(cell_descriptor):

            # Check simple case of initial cell state
            # Input descriptors are not cell inputs if they are inputs to init_cell_state_op
            if cell_descriptor.child_ops[-1].outputs[0] in descriptor.init_cell_state_op.inputs:
                return False
            else:
                output_shape = []
                output_ops = [op for op in cell_descriptor.child_ops if cell_descriptor.is_output_op(op)]
                if len(output_ops) > 0:
                    output_shape = converter_context.graph_helper.get_op_output_shape(output_ops[-1])

                # The output shape can be B,T if F is trivially 1. This happens in some TF models
                # which contain a reshape after the input data or
                # The output shape can T, F where B is trivially 1. This is seen in some models
                # with a single time step.
                if len(output_shape) >= 2 and output_shape[-1] == descriptor.time_steps() and output_shape[-2] == input_shape[1] or \
                        output_shape[-2] == descriptor.time_steps() and output_shape[-1] == input_shape[1]:
                    # We need to account for the corner case where initial hidden state and input data have the same shape
                    # In that case both the hidden input and data input descriptor will pass the above test
                    # In that case we assume descriptors are ordered i.e input data is the first input as a last resort
                    if input_shape == state_shape:
                        return cell_descriptor == input_descriptors[0]
                    else:
                        return True
                return False

        cell_input_descriptors = list(filter(is_cell_input_descriptor, input_descriptors))
        cell_state_descriptors = [d for d in input_descriptors if d not in cell_input_descriptors]

        # This is the case when distinct LSTM cells are stacked above each other
        is_stacked_above_cell = self.is_stacked_cell(input_descriptors)
        if not is_stacked_above_cell:
            # There can be at most one cell input descriptor in this case
            if len(list(cell_input_descriptors)) != 1:
                raise ConverterError('Unable to resolve LSTM input layer name.')

            input_layer_name = cell_input_descriptors[0].output_names[0]

            # need to reshape if the input to the cell is 2D and reshape input has been ignored
            if self._needs_reshape_to_restore_time_dimension(converter_context,
                                                             descriptor, cell_input_descriptors[0], input_shape):
                input_layer_name = self._add_reshape_to_restore_time_dimension(
                    ir_graph, descriptor, input_layer_name, input_shape)

        else:
            input_layer_name = input_descriptors[0].output_names[0]

        # This checks for sequential LSTM layers i.e the output of this LSTM descriptor is the input to another
        is_stacked_below_cell = self.is_stacked_cell(output_descriptors)
        descriptor.set_is_stacked_cell(is_stacked_below_cell)

        # User initial state determines if an initial state was passed and if a final cell state will be returned.
        if len(list(cell_state_descriptors)) == 2:
            # There are two cell state descriptors
            # Case 1: LSTM op returns the final and hidden state to be consumed by other ops,
            #         in this case we need user initial state to be true
            if descriptor.returns_state():
                user_initial_state = True
            else:
                # Case 2: Initial state is provided but is zero and final states are not consumed by other ops
                # TO-DO: remove once backend spec is aligned
                user_initial_state = all(d.value.any() for d in cell_state_descriptors
                                         if isinstance(d, ConstantLayerResolver.Descriptor))
        else:
            # Case 3: No initial state
            user_initial_state = False

        # At the very least each runtime, will return a buffer containing all time-steps
        output_names = [descriptor.rolled_cell_output_name]
        h_0_input_name, c_0_input_name = "", ""

        # if there is no user initial state then the initial hidden state and cell state inputs are empty
        # if the user initial state is set but it is a stacked cell, then the initial input is ignored anyway.
        # This is because the initial state for the stacked below cell is the output of the previous cell.
        # Else, the initial hidden state and cell state are expected to have predefined input names
        # and all possible output buffers are produced
        if not user_initial_state or is_stacked_below_cell and len(output_descriptors) > 0:
            # This does not change anything, but it is more illustrative to have it explicitly checked
            pass
        elif user_initial_state:
            h_0_input_name = cell_state_descriptors[0].output_names[0]
            c_0_input_name = cell_state_descriptors[1].output_names[0]
            # if the user initial state is provided, then the final cell state and hidden state are returned
            output_names.extend(descriptor.output_names)

        input_names = [input_layer_name]

        if h_0_input_name:
            input_names.append(h_0_input_name)
        if c_0_input_name:
            input_names.append(c_0_input_name)

        # The internal state of the cell is reset at each time step if there is no user initial state or
        # if the cell is stacked. Note if the cell is stacked, then the initial state is the previous
        # final cell state and hidden state at time step 0
        reset_at_time_step_0 = not user_initial_state or is_stacked_below_cell

        return ir_graph.add(LstmOp(name=descriptor.cell_0.child_ops[-1].name,
                                   input_weights=input_weights,
                                   gate_bias=gates_biases,
                                   hidden_state_weights=gates_weights,
                                   reset_state_at_time_step_0=reset_at_time_step_0,
                                   c_0_input_name=c_0_input_name,
                                   h_0_input_name=h_0_input_name,
                                   hidden_size=state_shape[1]), input_names=input_names, output_names=output_names)

    @staticmethod
    def _merge_concat_timestep_descriptor(converter_context, descriptor, output_descriptors):
        # This function merges the concat descriptor into cell_0 if one exists and
        # it only concatenates all time-steps

        lstm_concat_outputs = [d for d in output_descriptors if
                               isinstance(d, (ConcatLayerResolver.Descriptor, PackLayerResolver.Descriptor))]
        if len(lstm_concat_outputs) == 1:
            # ensure input descriptors to concat are all lstm ops
            lstm_concat_input_descriptors = converter_context.topology_resolver.get_input_layers_for(
                lstm_concat_outputs[0])
            # These need to be in order
            if lstm_concat_input_descriptors == descriptor.cell_0.unrolled_cells:
                # This could be either a concatenation of all time-steps or a concatenation of a single timestep's hidden state and cell state
                # We only support merging the former
                concat_input_names = [input_.name for input_ in lstm_concat_outputs[0].child_ops[-1].inputs]
                for i, desc in enumerate(lstm_concat_input_descriptors):
                    cell_state_out, _ = desc.output_names  # note that this will return cell_state, hidden_state at this stage

                    # check that cell state out is not one of the inputs to the concat op
                    # if it is, then we return instantly as we can only consume a concat that merges hidden states
                    # If the cell state input is not present then it is implicitly the hidden state
                    if cell_state_out in concat_input_names:
                        return
                converter_context.merge_descriptors(lstm_concat_outputs[0], descriptor.cell_0)

    @staticmethod
    def _ignore_split_reshape_layers(converter_context, descriptor, input_descriptors):
        # This function ignores the reshape and split layers if applicable provided timestep merging has occurred
        # Note that the reshape and split layers simply divide the feature data into time-steps, given merging
        # these layers can be safely ignored.
        reshape_descriptor = None
        split_descriptor = None
        lstm_split_inputs = [d for d in input_descriptors if
                             isinstance(d, (SliceLayerResolver.Descriptor, UnPackLayerResolver.Descriptor))]
        if len(lstm_split_inputs) == 1:
            lstm_split_out_descriptors = converter_context.topology_resolver.get_output_layers_for(lstm_split_inputs[0])
            # ensure output descriptors for split op are all the unrolled cells
            # These need not be in order
            if not list(set(lstm_split_out_descriptors) - set(descriptor.cell_0.unrolled_cells)):
                split_descriptor = lstm_split_inputs[0]
                try:
                    # This block of code searches for the (Placeholder, reshape) input patter to the lstm
                    # and removes the reshape if it simply squeezes the dimension
                    # note that by checking if the reshape shape is in the placeholder dimension
                    # we should be able to guarantee that all other dimensions must be 1
                    _, reshape_tensor = converter_context.graph_helper.get_op_input_tensors(
                        split_descriptor.child_ops[-1], ("Const", "Reshape"))
                    result = converter_context.graph_helper.get_op_sequence(reshape_tensor.op,
                                                                            ['Reshape', 'Placeholder'])
                    placeholder_tensor_shape = converter_context.graph_helper.get_op_output_shape(result[-1])
                    reshape_tensor_shape = converter_context.graph_helper.get_op_output_shape(result[-1])

                    if all(r_shape in placeholder_tensor_shape for r_shape in reshape_tensor_shape):
                        reshape_descriptor = converter_context.topology_resolver.get_input_layers_for(split_descriptor)[
                            0]
                        reshape_descriptor.set_ignored(True)
                except (OperationNotFoundError, TensorNotFoundError):
                    # this means the reshape merging failed, and that is no error so it should be skipped
                    pass
                # check if there is only one
                split_descriptor.set_ignored(True)

        return reshape_descriptor is not None and split_descriptor is not None

    @classmethod
    def is_stacked_cell(cls, descriptors):
        return len(descriptors) >= 1 and isinstance(descriptors[0], LstmLayerResolver.UnrolledTimeStepDescriptor)

    @classmethod
    def _needs_reshape_to_restore_time_dimension(cls, converter_context, cell_descriptor, in_descriptor, input_shape):

        if len(input_shape) != 2:
            return False

        input_shape = converter_context.graph_helper.get_op_output_shape(
            cell_descriptor.cell_input_concat_op.inputs[0].op)
        in_descriptor_shape = converter_context.graph_helper.get_op_output_shape(in_descriptor.child_ops[-1])[0]
        return in_descriptor_shape != [input_shape[0], cell_descriptor.time_steps(), input_shape[1]]

    @classmethod
    def _merge_unrolled_input_cells(cls, converter_context, input_descriptors, descriptor):
        lstm_inputs = [i for i in input_descriptors if isinstance(i, LstmLayerResolver.UnrolledTimeStepDescriptor)]
        unrolled_inputs = [d for d in lstm_inputs if descriptor.is_unrolled_cell_of(d.cell_0)]
        for input_descriptor in unrolled_inputs:
            converter_context.merge_descriptors(descriptor, input_descriptor.cell_0)
            input_descriptor.cell_0.unrolled_cells.append(descriptor)
            descriptor.cell_0 = input_descriptor.cell_0

    def _resolve_weights(self, descriptor, graph_helper, state_shape):
        merged_weights = graph_helper.evaluate_tensor_output(descriptor.gates_matmul_op.inputs[1])
        input_weights_slice_index = np.shape(merged_weights)[0] - state_shape[-1]
        weights_list = np.split(merged_weights,
                                indices_or_sections=[input_weights_slice_index],
                                axis=0)
        input_weights = weights_list[0]
        input_weights = self._reorder_tensorflow_gates_weights(input_weights)
        gates_weights = weights_list[1]
        gates_weights = self._reorder_tensorflow_gates_weights(gates_weights)
        return gates_weights, input_weights

    def _resolve_biases(self, descriptor, graph_helper):
        gates_biases = graph_helper.evaluate_tensor_output(descriptor.gates_biases_op.inputs[1])
        gates_biases = np.split(gates_biases, indices_or_sections=4, axis=0)
        self._add_scalar_to_gate_bias(self._TENSORFLOW_FORGET_GATE_INDEX, descriptor.forget_bias_value, gates_biases)
        gates_biases = self._reorder_tensorflow_gates_biases(gates_biases)
        return gates_biases

    @classmethod
    def _add_scalar_to_gate_bias(cls, gate_index, bias_value, gates_biases):
        gates_biases[gate_index] += bias_value

    @classmethod
    def _reorder_tensorflow_gates_weights(cls, weights):
        weights = np.split(weights, indices_or_sections=4, axis=1)
        reordered = [
            weights[cls._TENSORFLOW_INPUT_GATE_INDEX],
            weights[cls._TENSORFLOW_FORGET_GATE_INDEX],
            weights[cls._TENSORFLOW_OUTPUT_GATE_INDEX],
            weights[cls._TENSORFLOW_STATE_GATE_INDEX],
        ]
        return np.concatenate(reordered, axis=1)

    @classmethod
    def _reorder_tensorflow_gates_biases(cls, biases):
        reordered = [
            biases[cls._TENSORFLOW_INPUT_GATE_INDEX],
            biases[cls._TENSORFLOW_FORGET_GATE_INDEX],
            biases[cls._TENSORFLOW_OUTPUT_GATE_INDEX],
            biases[cls._TENSORFLOW_STATE_GATE_INDEX],
        ]
        return np.concatenate(reordered, axis=0)

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        # Merge the current descriptor into cell_0 if it appears to be a time-step of cell_0
        self._merge_unrolled_input_cells(converter_context, input_descriptors, descriptor)

        # if both reshape and split have been ignored then we may not need to reshape
        self._ignore_split_reshape_layers(converter_context, descriptor, input_descriptors)

        # Merge concat descriptor only if it concatenates all time-steps and there are no other LSTM layers left to roll
        if not any(isinstance(out_desc, LstmLayerResolver.UnrolledTimeStepDescriptor) for out_desc in output_descriptors):
            self._merge_concat_timestep_descriptor(converter_context, descriptor, output_descriptors)

        return
