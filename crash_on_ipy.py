#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#

"""Open ipdb if program crashes
Dependency:
  - ipdb

Usage:
  # Import it in your program and you are good to go.
  import crash_on_ipy

"""

import ipdb
import sys
import traceback


def info(type, value, tb):
  traceback.print_exception(type, value, tb)
  ipdb.pm()


sys.excepthook = info