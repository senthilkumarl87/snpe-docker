# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


import traceback
from abc import abstractmethod, ABC
import qti.aisw.converters.common.converter_ir.op_graph as op_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrder
from qti.aisw.converters.common.converter_ir.op_policies import ConversionNamePolicy
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.common_base import ConverterBase


class ConverterFrontend(ConverterBase, ABC):

    class ArgParser(ConverterBase.ArgParser):
        def __init__(self, **kwargs):
            super(ConverterFrontend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--out_node', type=str, action='append', default=[],
                                       help="Name of the graph\'s output nodes. Multiple output nodes "
                                            "should be provided separately like: \n"
                                            "    --out_node out_1 --out_node out_2")
            self.add_optional_argument('--input_type', "-t", nargs=2, action='append',
                                       help='Type of data expected by each input op/layer. Type for each input '
                                            'is |default| if not specified. For example: "data" image.Note that '
                                            'the quotes should always be included in order to handle special '
                                            'characters, spaces,etc. For multiple inputs specify multiple '
                                            '--input_type on the command line.\n'
                                            'Eg: \n'
                                            '   --input_type "data1" image --input_type "data2" opaque \n'
                                            'These options get used by DSP runtime and following descriptions '
                                            'state how input will be handled for each option.\n'
                                            'Image: \n'
                                            'Input is float between 0-255 and the input\'s mean is 0.0f '
                                            'and the input\'s max is 255.0f. We will cast the float to uint8ts '
                                            'and pass the uint8ts to the DSP. \n'
                                            'Default: \n'
                                            'Pass the input as floats to the dsp directly and the DSP '
                                            'will quantize it.\n'
                                            'Opaque: \n'
                                            'Assumes input is float because the consumer layer(i.e next '
                                            'layer) requires it as float, therefore it won\'t be quantized.\n'
                                            'Choices supported:\n   ' + '\n   '.join(op_graph.InputType.
                                                                                     get_supported_types()),
                                       metavar=('INPUT_NAME', 'INPUT_TYPE'), default=[])
            self.add_optional_argument('--input_encoding', "-e", nargs=2, action='append',
                                       help='Image encoding of the source images. Default is bgr. \n'
                                            'Eg usage: \n'
                                            '   "data" rgba \n'
                                            'Note the quotes should always be included in order to handle '
                                            'special characters, spaces, etc. For multiple '
                                            'inputs specify --input_encoding for each on the command line.\n'
                                            'Eg:\n'
                                            '    --input_encoding "data1" rgba --input_encoding "data2" other\n'
                                            'Use options:\n '
                                            'color encodings(bgr,rgb, nv21...) if input is image; \n'
                                            'time_series: for inputs of rnn models; \n'
                                            'other: if input doesn\'t follow above categories or is unknown. \n'
                                            'Choices supported:\n   ' + '\n   '.join(op_graph.InputEncodings.
                                                                                     get_supported_encodings()),
                                       metavar=('INPUT_NAME', 'INPUT_ENCODING'), default=[])
            self.add_optional_argument("--debug", type=int, nargs='?', default=-1,
                                       help="Run the converter in debug mode.")

    def __init__(self, args,
                 naming_policy=ConversionNamePolicy(),
                 shape_inference_policy=None,
                 axis_order=AxisOrder(),
                 custom_op_factory=None):
        super(ConverterFrontend, self).__init__(args)
        self.debug = args.debug
        if not self.debug:
            # If --debug provided without any argument, enable all the debug modes upto log_debug3
            self.debug = 3
        setup_logging(self.debug)

        self.output_nodes = args.out_node
        self.graph = op_graph.IROpGraph(naming_policy, shape_inference_policy,
                                        args.input_type, args.input_encoding, axis_order,
                                        output_nodes=self.output_nodes)

        self.custom_op_config_paths = args.custom_op_config_paths
        self.custom_op_factory = custom_op_factory

    @abstractmethod
    def convert(self):
        """
        Convert the input framework model to IROpGraph: to be overridden by each framework
        """
        pass

    # TODO: Move once converter base hierarchy is refactored
    def populate_custom_op_collection(self,
                                      model,
                                      converter_type='onnx',
                                      **kwargs):
        # Create a custom op collection based on configs provided by user
        if self.custom_op_config_paths is not None:
            for config_path in self.custom_op_config_paths:
                try:
                    self.custom_op_factory.parse_config(config_path,
                                                        model=model,
                                                        converter_type=converter_type,
                                                        **kwargs)
                except Exception as e:
                    if not is_log_level_debug():
                        traceback.print_exc()
                    log_error("Error populating custom ops from: {}\n {}".format(config_path,
                                                                                 str(e)))
                    sys.exit(-1)

                if not len(self.custom_op_factory.op_collection) and \
                        not self.custom_op_factory.default_op_collection:
                    raise LookupError("CUSTOM_OP_NOT_FOUND: "
                                      "None of the custom Ops present in the "
                                      "config were found in the provided model.")
