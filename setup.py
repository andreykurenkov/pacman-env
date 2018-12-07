#!/usr/bin/env python

from distutils.core import setup

setup(name='pacman-env',
      version='1.0',
      description='Pacman Env',
      install_requires=[
          'gym',
      ],
      author='Andrey Kurenkov',
      author_email='andreyk@stanford.edu',
      url='https://github.com/andreykurenkov/pacman-env',
      packages=['pacman_env'],
      package_dir={'': 'src'},
      package_data={'pacman_env': ['layouts/*.lay']},
     )
