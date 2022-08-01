"""Setup for wheel distribution. We only use this for colab. For everything else we use conda."""

import os
import setuptools

from setuptools import setup

# pip needs requirements here; keep in sync with meta.yaml!
requirements = [
    "numpy",
    "torch@https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp38-cp38-linux_x86_64.whl",
    "torchaudio==0.7.0", # somehow required, and also needs to match torch version (intentionally off by one major version)
    "click",
    "deprecated",
    "gitpython>=3.1",
    "h5py",
    "importlib_resources",
    "matplotlib",
    "pandas",
    "pytest",
    "pyyaml",
    "requests",
    "scipy",
    "seaborn==0.10",
    "scikit-image",
    "scikit-learn",
    "tensorboard",
    "tifffile",
    "tqdm",
    "torchvision",
    "opencv-python",
    "edt",
]

setup(
    name='decode',
    version='0.11.1',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'decode.train = decode.neuralfitter.train.train:main',
            'decode.fit = decode.neuralfitter.inference.infer:main',
            'decode.infer = decode.neuralfitter.inference.infer:main',
        ],
    },
    zip_safe=False,
    url='https://rieslab.de',
    license='GPL3',
    author='Lucas-Raphael Mueller',
    author_email='',
    description=''
)
