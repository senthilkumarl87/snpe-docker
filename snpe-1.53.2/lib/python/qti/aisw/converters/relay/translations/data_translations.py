# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.converter_ir.op_adapter import (
    ConcatOp,
    ConstantOp,
    GatherOp,
    NeuronOp,
    PermuteOp,
    ReshapeOp,
    ResizeOp,
    SliceOp,
    StridedSliceOp
)

from qti.aisw.converters.relay.translations.relay_translations import RelayTranslationBase
from qti.aisw.converters.relay.translations import RelayTranslations

import tvm
from tvm import relay
from tvm.relay.testing import run_infer_type


# ------------------------------------------------------------------------------
#   Clip
# ------------------------------------------------------------------------------
class RelayClipTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayClipTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["clip_min"] = relay_expr.attrs.a_min
        attr_dict["clip_max"] = relay_expr.attrs.a_max

        log_debug3("\tclip min {}", attr_dict["clip_min"])
        log_debug3("\tclip max {}", attr_dict["clip_max"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, NeuronOp.TRANSLATION_KEY)

        ir_op = NeuronOp(op_name,
                         neuron_type=NeuronOp.Type.RELU_MIN_MAX,
                         min_clamp=attr_dict["clip_min"],
                         max_clamp=attr_dict["clip_max"])
        return ir_op


RelayTranslations.register_translation(RelayClipTranslation(),
                                       converter_type('clip', 'relay'))


# ------------------------------------------------------------------------------
#   Concat
# ------------------------------------------------------------------------------
class RelayConcatTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayConcatTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["axis"] = int(relay_expr.attrs.axis)

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ConcatOp.TRANSLATION_KEY)

        ir_op = ConcatOp(op_name, attr_dict["axis"])
        return ir_op


RelayTranslations.register_translation(RelayConcatTranslation(),
                                       converter_type('concatenate', 'relay'))


# ------------------------------------------------------------------------------
#   ExpandDims
# ------------------------------------------------------------------------------
class RelayExpandDimsTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayExpandDimsTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        expand_dims_attrs = relay_expr.attrs
        attr_dict['axis'] = expand_dims_attrs.axis

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY)
        axis = attr_dict['axis']
        log_debug3("\taxis {}", axis)

        mod = tvm.IRModule.from_expr(relay_expr)
        mod = relay.transform.InferType()(mod)
        output_shape = mod["main"].ret_type.shape
        output_shape = np.array(output_shape)

        ir_op = ReshapeOp(op_name, output_shape=output_shape)
        return ir_op


RelayTranslations.register_translation(RelayExpandDimsTranslation(),
                                       converter_type('expand_dims', 'relay'))


# ------------------------------------------------------------------------------
#   Flatten
# ------------------------------------------------------------------------------
class RelayFlattenTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayFlattenTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        output_shape = list()
        output_shape.append(input_shape[0]) # batch
        output_shape.append(np.prod(input_shape[1:]))

        log_debug3("\tOp input shape {}", input_shape)
        log_debug3("\tOp new shape {}", output_shape)

        ir_op = ReshapeOp(op_name, output_shape)
        return ir_op


RelayTranslations.register_translation(RelayFlattenTranslation(),
                                       converter_type('batch_flatten', 'relay'))


# ------------------------------------------------------------------------------
#   Gather
# ------------------------------------------------------------------------------
class RelayGatherTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayGatherTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        gather_attrs = relay_expr.attrs
        attr_dict['axis'] = gather_attrs.axis

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, GatherOp.TRANSLATION_KEY)

        axis = attr_dict['axis']
        indices = relay_params[input_names[1]]

        if isinstance(indices, tvm.runtime.ndarray.NDArray) or isinstance(indices, tvm.runtime.NDArray):
            indices = indices.asnumpy()

        log_debug3("\tindices {}", indices)

        indices_output = input_names[1]
        quir_graph.add(ConstantOp(indices_output, indices), [], [indices_output])

        ir_op = GatherOp(op_name, axis=axis)
        return ir_op


RelayTranslations.register_translation(RelayGatherTranslation(),
                                       converter_type('take', 'relay'))


# ------------------------------------------------------------------------------
#   Reshape
# ------------------------------------------------------------------------------
class RelayReshapeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayReshapeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["new_shape"] = [int(val) for val in relay_expr.attrs.newshape]

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY)

        new_shape = attr_dict["new_shape"]
        log_debug3("\tReshape Op attribute new shape {}", new_shape)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        log_debug3("\tReshape Op input shape {}", input_shape)

        output_shape = run_infer_type(relay_expr).checked_type.shape
        output_shape = [int(x) for x in output_shape]
        log_debug3("\tReshape Op Calculated new shape {}", output_shape)

        ir_op = ReshapeOp(op_name, output_shape)
        return ir_op


RelayTranslations.register_translation(RelayReshapeTranslation(),
                                       converter_type('Reshape', 'relay'))


# ------------------------------------------------------------------------------
#   Resize
# ------------------------------------------------------------------------------
class RelayResizeTranslation(RelayTranslationBase):

    class TransformModes:
        ALIGN_CORNERS = "align_corners"
        ASYMMETRIC = "asymmetric"
        HALF_PIXEL = "half_pixel"

    class ScaleModes:
        BICUBIC = "bicubic"
        BILINEAR = "bilinear"
        NEAREST_NEIGHBOR = "nearest_neighbor"

    RELAY_CONSTS_TO_IR = {
        ScaleModes.BILINEAR: "bilinear",
        ScaleModes.NEAREST_NEIGHBOR: "nearest"
    }

    def __init__(self):
        super(RelayResizeTranslation, self).__init__()

    def extract_attributes(self, relay_expr: relay.expr.Call, relay_params: dict, **kwargs):
        attr_dict = {}
        resize_attrs = relay_expr.attrs

        attr_dict['size'] = [int(num) for num in getattr(resize_attrs, 'size')]
        attr_dict['layout'] = getattr(resize_attrs, 'layout', 'NCHW')

        log_debug3("\tsize {}", attr_dict['size'])
        log_debug3("\tlayout {}", attr_dict['layout'])

        output_dtype = getattr(resize_attrs, "output_dtype", None)
        if output_dtype is not None:
            raise ValueError("Unsupported conversion to output dtype {} for resize expr".format(output_dtype))

        scale_mode = getattr(resize_attrs, "method", self.ScaleModes.BILINEAR)
        if scale_mode == self.ScaleModes.BICUBIC:
            raise ValueError("Unsupported scale method bicubic for resize expr")
        attr_dict["resize_mode"] = self.RELAY_CONSTS_TO_IR[scale_mode]

        transform_mode = getattr(resize_attrs, "coordinate_transformation_mode", self.TransformModes.HALF_PIXEL)
        [attr_dict["align_corners"], attr_dict["half_pixel_centers"]] = False, False
        if transform_mode == self.TransformModes.ALIGN_CORNERS:
            attr_dict["align_corners"] = True
        elif transform_mode == self.TransformModes.HALF_PIXEL:
            attr_dict["half_pixel_centers"] = True

        log_debug3("\talign_corners {}", attr_dict["align_corners"])
        log_debug3("\thalf_pixel_centers {}", attr_dict["half_pixel_centers"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ResizeOp.TRANSLATION_KEY)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        if attr_dict['layout'] == 'NCHW':
            output_shape = [input_shape[0], input_shape[1], attr_dict['size'][0], attr_dict['size'][1]]
        elif attr_dict['layout'] == 'NHWC':
            output_shape = [input_shape[0], attr_dict['size'][0], attr_dict['size'][1], input_shape[3]]
        else:
            raise ValueError("Unknown data layout {}".format(attr_dict['layout']))

        ir_op = ResizeOp(op_name,
                         output_shape,
                         resize_mode=attr_dict["resize_mode"],
                         align_corners=attr_dict["align_corners"],
                         half_pixel_centers=attr_dict["half_pixel_centers"])
        return ir_op


RelayTranslations.register_translation(RelayResizeTranslation(),
                                       converter_type('resize', 'relay'))


# ------------------------------------------------------------------------------
#   Split
# ------------------------------------------------------------------------------
class RelaySplitTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySplitTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["axis"] = int(relay_expr.attrs.axis)
        attr_dict["slice_points"] = relay_expr.attrs.indices_or_sections

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, SliceOp.TRANSLATION_KEY)

        axis = attr_dict["axis"]
        slice_points = attr_dict["slice_points"]

        output_shapes = []
        slices = []
        num_outputs = 0

        input_shapes = converter_context.get_input_shapes(relay_expr)
        slice_input_shape = input_shapes[0][:]
        if isinstance(slice_points, tvm.ir.container.Array):
            log_debug3("\tslice points {}", slice_points)
            num_outputs = len(slice_points) + 1
            slices = [int(val) for val in slice_points]

            log_debug3("\tmax dim {}", slice_input_shape[axis])
            slice_sizes = [0] + slices + [slice_input_shape[axis]]
            log_debug3("\tslice sizes {}", slice_sizes)

            for i in range(num_outputs):
                output_shapes.append(slice_input_shape[:])
                output_shapes[i][axis] = slice_sizes[i + 1] - slice_sizes[i]
        elif isinstance(slice_points, tvm.tir.expr.IntImm):
            log_debug3("\tslice points {}", int(slice_points))
            num_outputs = int(slice_points)
            slices = [0] * (num_outputs - 1)

            for i in range(num_outputs):
                output_shapes.append(input_shapes[0][:])
                output_shapes[i][axis] = int(int(output_shapes[i][axis]) / num_outputs)
                if i > 0:
                    slices[i - 1] = output_shapes[i-1][axis]
        else:
            raise TypeError("Unsupported type {} for slice_points in SplitOp".format(type(slice_points)))

        log_debug3("\tnum_outputs {}", num_outputs)
        log_debug3("\tslices {}", slices)
        log_debug3("\toutput shapes {}", output_shapes)

        ir_op = SliceOp(op_name, axis=axis, slice_points=slices, output_shape=output_shapes)
        ir_op.num_outputs = num_outputs
        return ir_op


RelayTranslations.register_translation(RelaySplitTranslation(),
                                       converter_type('Split', 'relay'))


# ------------------------------------------------------------------------------
#   Squeeze
# ------------------------------------------------------------------------------
class RelaySqueezeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySqueezeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        squeeze_attrs = relay_expr.attrs
        attr_dict["axis"] = squeeze_attrs.axis

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        log_debug3("\tSqueeze Op input shape {}", input_shape)

        axis = attr_dict["axis"]
        log_debug3("\taxis {}", axis)

        if axis is None:
            output_shape = [dim for dim in input_shape if dim != 1]
        else:
            output_shape = []
            for index, shape in enumerate(input_shape):
                if index in axis:
                    if shape != 1:
                        raise ValueError("Input shape {} at axis {} should be 1", input_shape, index)
                    continue
                output_shape.append(shape)
        log_debug3("\tSqueeze Op new shape {}", output_shape)

        ir_op = ReshapeOp(op_name, output_shape)
        return ir_op


RelayTranslations.register_translation(RelaySqueezeTranslation(),
                                       converter_type('squeeze', 'relay'))


# ------------------------------------------------------------------------------
#   StridedSlice
# ------------------------------------------------------------------------------
class RelayStridedSliceTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayStridedSliceTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        strided_slice_attrs = relay_expr.attrs
        attr_dict['begin'] = strided_slice_attrs.begin
        attr_dict['end'] = strided_slice_attrs.end
        attr_dict['strides'] = strided_slice_attrs.strides
        attr_dict['slice_mode'] = strided_slice_attrs.slice_mode

        return  attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, StridedSliceOp.TRANSLATION_KEY)

        begin = attr_dict['begin']
        end = attr_dict['end']
        strides = attr_dict['strides']
        slice_mode = attr_dict['slice_mode']

        if slice_mode == 'size':
            raise ValueError("Unsupported slice mode {} in StridedSliceOp".format(slice_mode))

        if isinstance(begin, tvm.ir.container.Array):
            begin = [int(begin_points) for begin_points in begin]
        elif isinstance(slice_points, tvm.tir.expr.IntImm):
            begin = int(begin)
        else:
            raise TypeError("Unsupported type {} for begin in StridedSliceOp".format(type(begin)))

        if isinstance(end, tvm.ir.container.Array):
            end = [int(end_points) for end_points in end]
        elif isinstance(slice_points, tvm.tir.expr.IntImm):
            end = int(end)
        else:
            raise TypeError("Unsupported type {} for end in StridedSliceOp".format(type(end)))

        if isinstance(strides, tvm.ir.container.Array):
            strides = [int(strides_points) for strides_points in strides]
        elif isinstance(slice_points, tvm.tir.expr.IntImm):
            strides = int(strides)
        else:
            raise TypeError("Unsupported type {} for strides in StridedSliceOp".format(type(strides)))

        log_debug3("\tbegin {}", begin)
        log_debug3("\tend {}", end)
        log_debug3("\tstrides {}", strides)
        log_debug3("\tslice_mode {}", slice_mode)

        ir_op = StridedSliceOp(op_name, begin=begin, end=end, strides=strides)
        return ir_op


RelayTranslations.register_translation(RelayStridedSliceTranslation(),
                                       converter_type('strided_slice', 'relay'))


# ------------------------------------------------------------------------------
#   Transpose
# ------------------------------------------------------------------------------
class RelayTransposeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTransposeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        transpose_attr = relay_expr.attrs
        axes = transpose_attr.axes if hasattr(transpose_attr, 'axes') else None
        attr_dict['axes'] = axes

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PermuteOp.TRANSLATION_KEY)

        if attr_dict['axes'] is None:
            # reverse order if not specified
            input_shape = converter_context.get_input_shapes(relay_expr)[0]
            input_dimensions = len(input_shape)
            axes = [i for i in reversed(range(input_dimensions))]
        else:
            axes = [int(i) for i in attr_dict['axes']]

        log_debug3("\taxes {}", axes)

        return PermuteOp(op_name, axes)


RelayTranslations.register_translation(RelayTransposeTranslation(),
                                       converter_type('transpose', 'relay'))

