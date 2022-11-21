# ==============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
import traceback

from qti.aisw.converters.common.utils import code_to_message

try:
    import onnx
except ImportError as e:
    raise Exception(code_to_message.get_error_message("ERROR_ONNX_NOT_FOUND")(str(e), str(sys.path)))

from qti.aisw.converters.common.converter_ir import op_graph_optimizations, op_policies
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders
from qti.aisw.converters.common.converter_base import ConverterFrontend
from .util import *
from . import onnx_translations


# ------------------------------------------------------------------------------
#   The Converter Class
# ------------------------------------------------------------------------------
class OnnxConverterFrontend(ConverterFrontend):
    class ArgParser(ConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(OnnxConverterFrontend.ArgParser, self).__init__(**kwargs)
            # add command-line options custom to onnx converter
            self.add_optional_argument("--dry_run", type=str, nargs='?', const='info', default=None,
                                       help='Evaluates the model without actually converting any ops, and '
                                            'returns unsupported ops/attributes as well as unused inputs and/or '
                                            'outputs if any. Leave empty or specify "info" to see dry run as a '
                                            'table, or specify "debug" to show more detailed messages only"')
            self.add_optional_argument('-d', '--input_dim', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DIM'),
                                       help="The name and dimension of all the input buffers to the network specified in\n"
                                            "the format [input_name comma-separated-dimensions],\n"
                                            "for example: 'data' 1,224,224,3. \n"
                                            "Note that the quotes should always be included in order to handle special\n"
                                            "characters, spaces, etc.\n"
                                            "NOTE: This feature works only with Onnx 1.6.0 and above")
            self.add_optional_argument('-n', '--no_simplification', action='store_true', default=False,
                                       help="Do not attempt to simplify the model automatically. This may prevent some models from properly converting \n"
                                            "when sequences of unsupported static operations are present.")

    def __init__(self, args, *, custom_op_factory=None):
        super(OnnxConverterFrontend, self).__init__(args,
                                                    naming_policy=OnnxNamePolicy(),
                                                    shape_inference_policy=OnnxShapeInferencePolicy(),
                                                    axis_order=AxisOrders.ONNX,
                                                    custom_op_factory=custom_op_factory)
        self.translations = onnx_translations.OnnxTranslations
        self.dry_run = args.dry_run
        self.no_simplification = args.no_simplification
        self.op_info = onnx_translations.OpVersionInfo()
        if args.input_dim is not None:
            (in_node, in_dim) = list(zip(*args.input_dim))
            self.input_node = in_node
            self.input_dim = in_dim
        else:
            self.input_node = None
            self.input_dim = None

        # We can't run simplification and quantization overrides/custom ops as the simplification process
        # could possibly squash layers preventing the custom ops or quantization overrides from being used
        if not self.no_simplification and (args.quantization_overrides or args.custom_op_config_paths):
            self.no_simplification = True
            log_warning("Can't simplify the model when custom ops or quantization overrides are specified, converting without simplification.")

    def evaluate(self, model):
        """
        Performs a dry-run of the Onnx Model without actually converting it, highlighting potential issues with
        attributes, inputs/outputs or opset versions.
        :param model: An Onnx model
        :return:
        """
        from qti.aisw.converters.onnx import model_evaluator
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            log_warning("Potential errors found in {} as per Onnx's in-built checker tool".format(self.input_model_path))
            log_warning("{}: {}", type(e), str(e))
        log_info('Proceeding with model evaluation...................................\n')
        model_evaluator.setup_dry_run(model, self.dry_run)

    def convert(self):
        model = onnx.load(self.input_model_path)

        # Try to simplify the model first
        if not self.no_simplification:
            try:
                import onnxsim
                try:
                    model_optimized, check_ok = onnxsim.simplify(model)
                    if check_ok:
                        log_debug1("Successfully simplified the onnx model!")
                        model = model_optimized
                    else:
                        log_warning("Couldn't simplify the model, attempting normal conversion")
                except Exception as e:
                    log_warning("Onnx model simplification failed, trying unsimplified model. ({}: {})", type(e), str(e))
            except ImportError as e:
                log_warning("Couldn't import onnx-simplifier. ({}: {})", type(e), str(e))
                log_warning("Install the onnx-simplifier for better model compatibility: \"pip3 install onnx-simplifier\"")
            except Exception as e:
                log_warning("Unknown error ({}: {}) during import of onnx simplifier", type(e), str(e))

        self.op_info.set_global_op_ver(model)

        if self.dry_run:
            self.evaluate(model)
            sys.exit(0)

        if self.input_dim and self.input_node:
            self._update_input_node(model.graph)

        self.graph.weights = WeightProvider(model)
        self.graph.tensor_to_np_dtype = self._track_tensor_type(model.graph)

        # check to give priority to custom output node provided as argument over graph output
        if not self.output_nodes and model.graph.output:
            for value_info in model.graph.output:
                self.graph.output_nodes.append(str(value_info.name))

        # populate custom op nodes if config paths are provided; condition is checked in function
        self.populate_custom_op_collection(model, 'onnx')

        # extract inputs
        parameter_names = set()
        for tensor in model.graph.initializer:
            parameter_names.add(str(tensor.name))

        for value_info in model.graph.input:
            name = str(value_info.name)
            if name in parameter_names:
                # weights are usually listed as inputs too.
                continue
            self.translations.apply_method_to_op(converter_type("input", "onnx"),
                                                 onnx_translations.OnnxTranslationBase.ADD_INPUT_OP, value_info, self.graph)

        # extract parameters, infer shapes, etc.
        for i, src_op in enumerate(model.graph.node):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, src_op.op_type))
            src_type = converter_type(src_op.op_type, "onnx")

            try:
                # first check if layer is a registered custom op in an op collection.
                # If so, the layer is added and the outer loop continues.
                if self.custom_op_factory and src_op.op_type in self.custom_op_factory.op_collection:
                    src_type = converter_type('custom', "onnx")
                    node = self.translations.apply_method_to_op(src_type,
                                                                onnx_translations.OnnxTranslationBase.ADD_OP,
                                                                src_op,
                                                                self.graph)
                    self.graph.add_src_op_info(node.op.name, [i for i in src_op.input], [o for o in src_op.output])

                else:
                    # If the op is not a custom operation, check the version and use the
                    # native converter translation
                    supported_version = self.translations.apply_method_to_op(src_type,
                                                                             onnx_translations.OnnxTranslationBase.SUPPORTED_VERSION,
                                                                             src_op.op_type)
                    self.op_info.validate_op_ver(src_op, supported_version)

                    self.translations.apply_method_to_op(src_type,
                                                         onnx_translations.OnnxTranslationBase.ADD_OP,
                                                         src_op,
                                                         self.graph)
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                log_error("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

        return self.graph

    def _track_tensor_type(self, graph):
        tensor_to_np_dtype = {}

        for value_info in graph.input:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        for value_info in graph.value_info:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        for value_info in graph.output:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        return tensor_to_np_dtype

    def _update_input_node(self, graph):
        if onnx.version.version < '1.6.0':
            raise ValueError("--input_dim command supported with onnx versions >= 1.6.0")
        input_node = list(self.input_node)
        input_dim = list(self.input_dim)
        initializers = [node.name for node in graph.initializer]
        original_inputs = {node.name : node for node in graph.input}
        new_inputs = { name : dim for name, dim in zip(input_node, input_dim)}

        # At first, remove original graph inputs
        for node_name in original_inputs:
            if node_name not in initializers:
                graph.input.remove(original_inputs[node_name])

        # Case 1: --input_dim in graph inputs
        for name in new_inputs:
            if name in initializers:
                raise ValueError("--input_dim command not supported with initializer " + name)
            elif name in original_inputs:
                dim = new_inputs[name]
                dims = tuple(map(int, dim.split(',')))
                input_new = onnx.helper.make_tensor_value_info(name,
                    onnx.TensorProto.FLOAT, dims)
                graph.input.insert(0, input_new)
                input_node.remove(name)
                input_dim.remove(dim)
            else:
                continue
        if len(input_node) == 0 and len(input_dim) == 0:
            return

        # Case 2: --input_dim in graph nodes
        bufs_to_remove = set()
        for i, src_op in enumerate(graph.node):
            for output_buf_name in src_op.output:
                if output_buf_name in input_node:
                    position = input_node.index(output_buf_name)
                    dim = input_dim[position]
                    dims = tuple(map(int, dim.split(',')))
                    input_new = onnx.helper.make_tensor_value_info(output_buf_name,
                        onnx.TensorProto.FLOAT, dims)
                    graph.input.insert(0, input_new)
                    bufs_to_remove.add(output_buf_name)
                    input_node.remove(output_buf_name)
                    input_dim.remove(dim)
        if len(input_node) != 0 and len(input_dim) != 0:
            invalid_names = ", ".join(input_node)
            raise ValueError("--input_dim command input name(s) not found: {}".format(invalid_names))

        # Cleaning all the nodes before the input node
        nodes_to_remove = []
        while bufs_to_remove:
            node_name = bufs_to_remove.pop()
            node_list = [node for node in graph.node if (node.name == node_name or node_name in node.output)]
            if not node_list:
                continue
            node = node_list[0]
            bufs_to_remove.update(set(node.input))
            if node not in nodes_to_remove:
                nodes_to_remove.append(node)

        remaining_nodes = [node for node in graph.node if node not in nodes_to_remove]

        # Throw error when all buffers in a branch are not specified
        for node in nodes_to_remove:
            for output in node.output:
                for remaining_node in remaining_nodes:
                    if output in remaining_node.input and output not in self.input_node:
                        raise ValueError("Cannot disconnect node with outputs: {} as output buffer"
                                         ": {} is still in use and was not specified".format
                                         (str(node.output), str(output)))
            graph.node.remove(node)


# ------------------------------------------------------------------------------
#   Policies
# ------------------------------------------------------------------------------
class OnnxNamePolicy(op_policies.ConversionNamePolicy):
    def __init__(self):
        op_policies.ConversionNamePolicy.__init__(self)

    def get_op_name(self, op):
        count = self.type_count.get(op.type, 0)
        self.type_count[op.type] = count + 1
        if op.name:
            return str(op.name)
        elif op.type == 'custom':
            return "%s_%s_%d" % (str(op.custom_type).lower(), op.type, count)
        else:
            return "%s_%d" % (op.type, count)


class OnnxShapeInferencePolicy(op_policies.ConversionShapeInferencePolicy):

    def infer_shape(self, op, input_shapes):
        return onnx_translations.OnnxTranslations.apply_method_to_op(op.type,
                                                                     onnx_translations.OnnxTranslationBase.INFER_SHAPE,
                                                                     op,
                                                                     input_shapes)
