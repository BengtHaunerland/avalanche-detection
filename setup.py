from setuptools import find_packages, setup

setup(
	name='avalanche-detection',
	version='0.0.1',
	packages=find_packages(),
	include_package_data=True,
	package_data={'gui': ['segmentation_gui_config.yaml']},
	entry_points={'console_scripts': ['avalanche-seg-gui=gui.segmentation_gui:main']},
)
