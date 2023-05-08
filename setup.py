import os
from setuptools import setup, find_packages

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'shredder',
    'version.py',
)

setup(
    name='shredder',
    version='1.0.0',
    description='Astronomical image deblender',
    license='GPL',
    author='Erin Scott Sheldon',
    author_email='erin.sheldon@gmail.com',
    packages=find_packages(),
)
