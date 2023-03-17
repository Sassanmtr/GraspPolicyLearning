"""Install script for setuptools."""

import setuptools
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="fmm_control_lite",
    version="0.0.1",
    author="Eugenio Chisari",
    author_email="chisari@cs.uni-freiburg.de",
    install_requires=[
        "setuptools",
        "numpy",
        "spatialmath-python",
        "roboticstoolbox-python",
        "qpsolvers",
        "osqp"
    ],
    description="Smart control of FMM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # license="MIT",
    # url="https://github.com/chisarie/jax-agents",
    packages=setuptools.find_packages(),
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires=">=3.6",
)