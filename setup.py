from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['docopt', 'pandas', 'keras', 'tensorflow','opencv-python','tqdm','pathlib','torch','torchvision']

setup(
    name='Deepfake-challenge',
    version='0.3',
    author = 'Juan Carlos Vargas',
    author_email = 'jcarvargtz@hotmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="training setup, data download, and procesing for the kaggle's Deepfake detection challenge")

