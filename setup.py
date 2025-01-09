from setuptools import setup, find_packages

setup(
    name='PLANiT_PPA_utils',  # Name of your package
    version='0.1.0',  # Initial version
    description='A suite of utilities for cost analysis, KEPCO data processing, and solar GHI data handling.',
    author='Ssanghyun Hong',
    author_email='sanghyun@planit.institute',
    url='https://github.com/PLANiT-Institute/pyPPA.git',  # Replace with your GitHub repo URL
    packages=find_packages(include=['PLANiT_PPA_utils', 'PLANiT_PPA_utils.*']),  # Include the `PLANiT_PPA_utils` directory
    install_requires=[
        'geopandas',
        'pandas',
        'numpy',
        'tqdm',
        'plotly',
        'requests',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)



