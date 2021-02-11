from setuptools import setup, find_packages

import os
package_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qttk')
version_filepath = os.path.join(package_dir, 'version.txt')
with open(version_filepath) as file_ptr:
    version = file_ptr.read().strip()

git_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(git_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='qttk',
    version=version,    
    description='Quantitative Trading Toolkit',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='https://github.com/conlan-scientific/qttk',
    author='Conlan Scientific Open-source Development Cohort',
    author_email='chris@conlan.io',
    license='GNU General Public License v3.0',
    packages=find_packages(),
    include_package_data=True,
    # package_data={
    #     '' : [
    #         'qttk/*.txt',
    #         'data/*.csv',
    #         'data/eod/*.csv',
    #         'data/alternative_data/*.csv',
    #         'data/validation_data/*.csv',
    #     ],
    # },
    install_requires=[
        'pandas>=1.1.5',
        'numpy>=1.8.5',
        'matplotlib>=3.1.3',
        'psutil>=5.7.3',  
        'scipy>=1.5.4',             
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

