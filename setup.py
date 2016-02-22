from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name='somlib',
    version='0.1.0',
    description='Self-Organizing Map Python implementation',
    long_description=long_description,
    url='https://github.com/nmarincic/somlib',
    author='Nikola Marincic',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5.1',
    ],
    keywords='som self-organizing map library',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['numpy','pandas','matplotlib'],
    entry_points={
        'console_scripts': [
            'somlib=somlib.__main__:main',
        ],
    },
)