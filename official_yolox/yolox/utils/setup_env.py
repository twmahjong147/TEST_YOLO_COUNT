#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import os
import cv2

__all__ = ["configure_module"]


def configure_module(ulimit_value=8192):
    """Configure pytorch module environment."""
    try:
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, rlimit[1]))
    except Exception:
        pass

    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
