import setuptools
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version_line = open("radar_scenes/_version.py", "rt").read()
reg_ex_version = r"^__version__ = ['\"]([^'\"]*)['\"]"
res = re.search(reg_ex_version, version_line, re.M)
verstr = res.group(1)

install_requires = [
    "Pillow>=8.1.0",
    "PySide2>=5.15.2",
    "QDarkStyle>=2.8.1",
    "h5py>=3.1.0",
    "matplotlib>=3.3.3, <3.4.0", # restricted to allow usage with python 3.6
    "numpy>=1.19.4, <1.20.0",  # restricted to allow usage with python 3.6
    "pip>=19.0.3",
    "pyqtgraph>=0.11.1",
    "scipy>=1.5.4, <1.6.0",  # restricted to allow usage with python 3.6
    "setuptools>=40.8.0"
]

tests_require = [

]

setuptools.setup(
    name="radar_scenes",
    version=verstr,
    author="Ole Schumann",
    author_email="hello@radar-scenes.com",
    maintainer="Ole Schumann",
    description="Helper Package for the RadarScenes Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oleschum/radar_scenes",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    tests_require=tests_require,
    package_data={'': ['car.png']},
    keywords=["radar", "classification", "automotive", "machine learning"],
    entry_points={
        'gui_scripts': [
            'rad_viewer = radar_scenes.viewer:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
