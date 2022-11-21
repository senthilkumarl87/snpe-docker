# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from math import isclose
from .converter_utils import *
from . import code_to_message


def pads_symmetric(pads):
    num_dims = len(pads)//2
    for i in range(num_dims):
        if pads[i] != pads[i+num_dims]:
            return False
    return True


def pads_righthanded(pads):
    num_dims = len(pads)//2
    for i in range(num_dims):
        if pads[i] != 0:
            return False
    # don't call all zeros right-handed
    return not all(x == 0 for x in pads)


def get_broadcasted_shape(input_shapes: list, axis_order) -> list:
    """
    Helper function to use in cases where input shapes differ while performing arithmetic operations
    (e.g. input_shapes: [[64], [1,255,255,64]], result: [1,255,255,64])
    :param input_shapes: list of shapes to broadcast
    :param axis_order: the AxisOrder object of the src fw. Used to determine whether shapes
                       should be broadcasted IR axis ordering or src fw
    :return: the broadcasted output shape
    :raises: ValueError if any shape in the input_shapes list are not broadcastable with eachother
    """
    def _are_inputs_broadcastable(input_shapes_):
        for i in range(len(input_shapes_) - 1):
            if any(not broadcastable(input_shapes_[i], input_shapes_[j]) for j in range(i + 1, len(input_shapes_))):
                return False
        return True

    def _get_output_shape(input_shapes_):
        # Uses numpy function to calculate shape, numpy throws shape mismatch errors
        inputs = [np.zeros(input_shape, dtype=np.byte) for input_shape in input_shapes_]
        try:
            output_shape_ = list(np.broadcast(*inputs).shape)
        except ValueError:
            raise ValueError("Shape mismatch, {} cannot be broadcast to a single shape".format(input_shapes_))

        return output_shape_

    if len(input_shapes) == 1:
        return input_shapes[0]

    input_ir_shapes = [axis_order.permute_shape_to_ir(shape) for shape in input_shapes]
    inputs_broadcastable = _are_inputs_broadcastable(input_ir_shapes)
    if inputs_broadcastable:  # first check if axis change to ir is broadcastable
        return axis_order.permute_shape_from_ir(_get_output_shape(input_ir_shapes))
    return _get_output_shape(input_shapes)


# -----------------------------
# Util for common chain replace
# -----------------------------
def chain_matched_eltwise_ops(graph, matched_node_list, op_class):
    """
    Replaces matched elementwise op nodes that have > 2 inputs with a chain of binary elementwise nodes
    :param graph: The IROpGraph object
    :param matched_node_list: The list of nodes that are product or sum elementwise ops with > 2 inputs
    :param op_class: A reference to the op_adapter class for the elementwise operation we're replacing
    """
    for nodes_tuple in matched_node_list:
        old_node = nodes_tuple[0]
        old_node_name = old_node.op.name
        old_node_inputs = old_node.input_names
        current_idx = graph.list_nodes().index(old_node)
        old_node_output_buffers = graph.get_output_buffers(old_node)

        log_debug(code_to_message.get_debugging_message("DEBUG_ELEMENTWISEBINARY_CHAIN")
                  (old_node_name, old_node.op.type, old_node_inputs))

        # Remove the existing node with > 2 inputs
        graph.prune(old_node, force_remove=True)

        # Construct the names of nodes and input/output buffers in chain
        new_nodes_names = ["%s_chain_0" % old_node_name]
        new_nodes_input_buf_names = [[old_node_inputs[0], old_node_inputs[1]]]
        for i in range(1, len(old_node_inputs) - 1):
            new_nodes_input_buf_names.append([new_nodes_names[i-1], old_node_inputs[i+1]])
            new_nodes_names.append("%s_chain_%d" % (old_node_name, i))
        # Reusing new_nodes_names as new_nodes_output_buf_names for readability
        new_nodes_output_buf_names = list(new_nodes_names)

        # Chain's last output buffer must have the name of the original node's output buffer
        new_nodes_output_buf_names[-1] = old_node_output_buffers[0].name
        # The last node in chain must have the original node name
        new_nodes_names[-1] = old_node_name
        # Add constructed nodes into op_graph
        for i in range(len(new_nodes_names)):
            new_op = op_class(new_nodes_names[i])
            new_node = graph.add(new_op, new_nodes_input_buf_names[i], new_nodes_output_buf_names[i], idx=current_idx)
            # Returned node allows capturing of name assigned by naming policy
            new_nodes_names[i] = new_node.op.name
            new_nodes_output_buf_names[i] = new_node.output_names[0]
            current_idx += 1

        # Set input buffers of original sequence's consumers to the output buffer of last producer in new chain
        for i in range(len(old_node_output_buffers)):
            consumers = old_node_output_buffers[i].consumers
            for consumer in consumers:
                consumer.input_names.append(new_nodes_output_buf_names[-1])


# -------------------------------------------------------
# General
# -------------------------------------------------------
def expand_to_rank(shape, rank):
    """
    :type shape: list[int]
    :type rank: int
    :rtype: list[int]
    """
    result = shape[:]
    while len(result) < rank:
        result.insert(0, 1)
    return result


def to_list(val):
    if not val:
        return []
    if type(val) != list:
        return [val]
    return val


def broadcastable(shape1, shape2):
    """
    Checks if two shapes are can be broadcast into one another in the numpy sense.
    :param shape1: Shape of the data1
    :param shape2: Shape of the data2
    :return: boolean if broadcast is possible otherwise false
    """

    # loop backwards on both shapes and validate each index for broadcasting.
    # Eg: for [4,11,1,9] with [8,9], we only need to validate 8 and 9.
    for shape_idx1, shape_idx2 in zip(shape1[::-1], shape2[::-1]):
        if shape_idx1 != 1 and shape_idx2 != 1 and shape_idx1 != shape_idx2:
            return False
    return True


def compare_values(val1, val2, rtol=1.e-5, atol=1.e-6):
    """
    :param val1: type: (str, float, int, ndarray, list, set, dict)
    :param val2: type: (str, float, int, ndarray, list, set, dict)
    :param rtol: type: float The relative tolerance parameter to use if vals are numeric.
    :param atol: type: float The absolute tolerance parameter to use if vals are numeric.
    :return:
    """
    if type(val1) != type(val2):
        return False
    if type(val1) == list and type(val2) == list or \
            (type(val1) == set and type(val2) == set):
        if len(val1) != len(val2):
            return False
        return all([compare_values(i, j) for i, j in zip(val1, val2)])
    elif type(val1) == dict and type(val2) == dict:
        return all(val1_key in val2 and compare_values(val1_val, val2[val1_key])
                   for val1_key, val1_val in val1.items())
    elif type(val1 != val2) is np.ndarray:
        # Check if any value in arrays are different. Need shape check first since numpy.allclose
        # broadcasts if shapes are not equal
        return val1.shape == val2.shape and np.allclose(val1, val2, rtol=rtol, atol=atol)
    else:
        if type(val1) == float and type(val2) == float:
            # do tolerance comparison for floats
            return isclose(val1, val2, rel_tol=rtol, abs_tol=atol)
        return val1 == val2



