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
    version=__version__,
    url="https://github.com/esheldon/shredder",
    description='An astronomical image deblender',
    license='GPL',
    author='Erin Scott Sheldon',
    author_email='erin.sheldon@gmail.com',
    packages=find_packages(),
)
