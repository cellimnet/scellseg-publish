import setuptools
from setuptools import setup

install_deps = ['numpy', 'scipy', 'natsort',
                'tifffile', 'tqdm', 'numba', 
                'torch>=1.6',
                'opencv-python-headless']

try:
    import torch
    a = torch.ones(2, 3)
    version = int(torch.__version__[2])
    if version >= 6:
        install_deps.remove('torch')
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="scellseg",
    license="BSD-3-Clause License",
    author="Dejin Xun & Deheng Chen",
    author_email="xundejin@zju.edu.cn",
    description="a specializable cell segmentation tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cellimnet/scellseg-publish",
    setup_requires=[
      'pytest-runner',
      'setuptools_scm',
    ],
    packages=setuptools.find_packages(),
    use_scm_version=True,
    install_requires = install_deps,
    tests_require=[
      'pytest'
    ],
    extras_require = {
      'docs': [
        'sphinx>=3.0',
        'sphinxcontrib-apidoc',
        'sphinx_rtd_theme',
      ],
      'guis': [
        'pyqtgraph==0.11.0rc0', 
        'pyqt5', 
        ]
    },
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause License",
        "Operating System :: OS Independent",
    ),
     entry_points = {
        'console_scripts': [
          'scellseg = scellseg.__main__:main']
     }
)
