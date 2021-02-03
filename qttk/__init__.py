import qttk.indicators
import qttk.utils

import os
package_dir = os.path.dirname(os.path.abspath(__file__))
version_filepath = os.path.join(package_dir, 'version.txt')

with open(version_filepath) as file_ptr:
	version = file_ptr.read().strip()
	
__version__ = version
__author__ = 'Conlan Scientific Open-source Development Cohort'
__credits__ = 'Conlan Scientific'

