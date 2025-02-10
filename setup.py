from setuptools import setup, find_packages

setup(
    name="dasn2n",
    version="0.2.0",
    description="dasn2n: A Python package for denoising DAS data with Noise2Noise",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sachalapins/dasn2n",
    license="GPL-3.0-or-later",
    packages=find_packages(exclude=["examples", "data", "build", "dist"]),
    package_data={
        "dasn2n": ["weights/*"],
    },
    include_package_data=True,
    install_requires=[
        "torch>=1.13.1",
        "numpy>=1.23.5",
        "scikit-image>=0.18.0"
    ],
    extras_require={
        "optional": ["obspy", "scipy", "pandas"],
        "jupyter": ["jupyterlab", "ipython"]
    },
    classifiers=[
        "Development Status :: 4 - Beta"
        "Intended Audience :: Science/Research"
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.8, <3.13', # PyTorch currently recommends Python 3.8 - 3.12
)
