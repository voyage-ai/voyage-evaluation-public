"""
Processor classes
"""

import io
import math
import base64
import logging
from typing import List

logger = logging.getLogger(__name__)


class Processor:

    def __init__(self, model_name: str = None):
        pass

    def __call__(self, *args, **kwargs):
        """Process data for the model, and return a dictionary of inputs."""
        raise NotImplementedError
