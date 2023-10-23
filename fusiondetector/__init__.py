__version__ = '1.0.0'

import os
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(PACKAGE_DIR, 'artifacts')
IMAGES_DIR = os.path.join(PACKAGE_DIR, 'images')
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'outputs')
