from setuptools import setup, find_packages
from openevolve import __version__

setup(
    name="openevolve",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
)
