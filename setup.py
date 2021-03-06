from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['docopt', 'pandas', 'keras', 'tensorflow','opencv-python','tqdm','pathlib','torch','torchvision']


setup(
    name='trainer',
    version='1.6',
    author = 'Juan Carlos Vargas',
    author_email = 'jcarvargtz@hotmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    package_data={'trainer':[r'trainer/*.pth',r'trainer/*.npy',r'trainer/*.sh']},
    description="training setup, data download, and procesing for the kaggle's Deepfake detection challenge")

