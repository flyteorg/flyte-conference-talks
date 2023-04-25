from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [x.strip() for x in f.readlines()]

setup(
    name='pydata-seattle-2023-tutorial',
    version='0.0.0',
    url='https://github.com/flyteorg/flyte-conference-talks',
    author='Eduardo Apolinario',
    packages=find_packages(),    
    install_requires=install_requires,
)
