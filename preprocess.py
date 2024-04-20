import argparse
import glob
import json
import os
import pickle

import c3d
import numpy as np
from natsort import natsorted

from libs.utils import Camera, ensure_homogeneous, to_cartesian, to_homogeneous
