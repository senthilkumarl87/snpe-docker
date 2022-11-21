# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
from importlib.abc import MetaPathFinder, Loader
import importlib

# Extension exceptions to the qti.tvm import rules
lib_exts = ['relay', 'frontend', 'tflite']

class MyLoader(Loader):
    def module_repr(self, my_module):
        return repr(my_module)

    def load_module(self, fullname):
        old_name = fullname
        top_level_name = fullname.split(".")[0]

        # Two cases here:
        #   1. Force the tvm tflite import to use the qti tvm tflite import
        #   2. Redirect non tflite imports back to the real tvm
        if fullname == 'tvm.relay.frontend.tflite': # or fullname == '.tflite':
            fullname = 'qti.tvm.relay.frontend.tflite'
        elif fullname.startswith('qti.tvm') and not fullname.endswith('.tflite'):
            fullname = fullname.replace('qti.', '')

        # Always override tvm.nd due to some funky __init__ logic in tvm
        if fullname == 'tvm.nd':
            fullname = 'tvm.runtime.ndarray'

        my_module = importlib.import_module(fullname)
        # update/add import of both old and new name to point to the new module just imported
        sys.modules[old_name] = my_module
        sys.modules[fullname] = my_module
        return my_module

class MyImport(MetaPathFinder):
    def find_module(self, fullname, path=None):
        if  fullname == 'tvm.relay.frontend.tflite' or \
           (fullname.startswith('qti.tvm') and not fullname.endswith(tuple(lib_exts))):
            return MyLoader()
        return None

# overwrite all submodule imports.
sys.meta_path.insert(0, MyImport())
