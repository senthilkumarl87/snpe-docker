# =============================================================================
#
#  Copyright (c) 2017-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os


def check_validity(resource, *, is_path=False, is_directory=False, must_exist=True, extensions=[]):
    resource_path = os.path.abspath(resource)
    if is_path and os.path.isdir(resource_path):
        # For the case that resource path can be either dir or file
        is_directory = True
    if must_exist and not os.path.exists(resource_path):
        raise IOError('{} does not exist'.format(str(resource)))
    elif not is_directory:
        if os.path.exists(resource_path) and not os.path.isfile(resource_path):
            raise IOError('{} is not a valid {} file'.format(str(resource), str(extensions)))
        if extensions and \
                not any([os.path.splitext(resource_path)[1] == str(extension) for extension in extensions]):
            raise IOError("{} is not a valid file extension: {}".format(resource, str(extensions)))
    else:
        if os.path.exists(resource_path) and not os.path.isdir(resource_path):
            raise IOError('{} is not a valid directory'.format(str(resource)))
        elif extensions:
            raise ValueError("Directories cannot have a file extension".format(str(resource)))