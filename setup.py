from setuptools import setup

setup(
    name="isys_optitrack",
    version="0.0.52",
    author="Peter Somers",
    author_email="peter.somers@isys.uni-stuttgart.de",
    description="Tracking stuff with opti-track",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    packages=['isys_optitrack'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=['scipy',
                      'pandas',
                      'numpy',
                      'blender-stubs',
                      'opencv-python',
                      'tqdm',
                      'Pillow',
                      'natnetclient',
                      'witmotion']
)
