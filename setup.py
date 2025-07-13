from setuptools import setup, find_packages
import os
import sys

# Add the package directory to sys.path to import version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openevolve'))
from _version import __version__

setup(
    name="openevolve",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
)
