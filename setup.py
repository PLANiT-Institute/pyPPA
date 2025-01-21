from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt", 'rb') as f:
    requirements = f.read().splitlines()

setup(
    name="PLANiT_PPA",  # Replace with your package name
    version="1.0",
    description="PPA model for South Korea, developed by Sanghyun Hong (PLANiT)",
    author="Sanghyun Hong",  # Replace with your name or organization
    license="GPL-3.0",
    packages=find_packages(),  # Automatically find all packages
    install_requires=requirements,  # Load dependencies from requirements.txt
    python_requires=">=3.6",  # Specify the required Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)