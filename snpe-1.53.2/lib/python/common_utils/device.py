#
# Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

from .protocol import protocol
from .env_helper import env_helper
from os import remove, path, listdir
import logging

logger = logging.getLogger(__name__)

class Device:
    def __init__(self, device_id, archs, device_root, bridge="adb"):
        self.device_id = device_id
        if bridge == "adb":
            self.device_helper = protocol('adb', device_id)
        else:
            self.device_helper = protocol(device_id)
        self.env_helper = None
        self.device_root = device_root
        self.soc = self.get_soc()
        self.archs = archs

    def init_env(self, artifacts_dir, is_sdk):
        self.env_helper = env_helper(self.device_helper, self.archs, artifacts_dir, self.device_root, is_sdk)

    def setup_device(self):
        self.env_helper.init()

    def push_dir_data(self, src, dst):
        if path.isdir(src):
            for file in listdir(src):
                self.push_data(path.join(src, file), dst)

    def push_data(self, src, dst):
        ret, _, err = self.device_helper.push(src, dst)
        if ret != 0:
            logger.error('[{}] Failed to push: {}'.format(self.device_id, src))
            logger.error('[{}] stderr: {}'.format(self.device_id, err))

    def pull_data(self, src, dst):
        ret, _, err = self.device_helper.pull(src, dst)
        if ret != 0:
            logger.error('[{}] Failed to pull: {}'.format(self.device_id, src))
            logger.error('[{}] stderr: {}'.format(self.device_id, err))

    def get_soc(self):
        soc = self.device_helper.get_soc()
        return soc

    def __str__(self):
        return '{}-{}'.format(self.soc, self.device_id)
