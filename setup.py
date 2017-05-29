"""Setup script for lie_grp_diffeo.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages

setup(
    name='lie_grp_diffeo',

    version='0.1.0',

    description='Lie Group Diffeomorphisms',

    url='https://github.com/adler-j/lie_grp_diffeo',

    author='Jonas Adler',
    author_email='jonasadl@kth.se',

    license='GPLv3+',

    keywords='research development emission prototyping imaging tomography',

    packages=find_packages(exclude=['*test*']),
    package_dir={'lie_grp_diffeo': 'lie_grp_diffeo'},

    install_requires=['numpy', 'odl>=0.6']
)