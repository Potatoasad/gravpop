# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='gravpop',
    version='0.1.1',
    description="""Package to perform astrophysical population modelling using gravitational waves, 
                   specifically for modelling using Truncated Gaussian Mixtures or a hybrid scheme 
                   -large portions of this code are structured and computed very similar to gwpop""",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Asad Hussain',
    author_email='asadh@utexas.edu',
    url='https://github.com/potatoasad/gravpop',
    packages=find_packages(exclude=('tests', 'docs', 'dev'))
)
