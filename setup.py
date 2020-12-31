import os
import sys
from setuptools import setup


setup(name='tvn',
      version='1.0',
      description='Tiny Video Networks',
      author='George Kasparyants',
      author_email='gkasparyants@gmail.com',
      url='https://github.com/DELTA37/TVN.git',
      packages=['tvn',
                'tvn.cg',
                'tvn.non_local',
                'tvn.se',
                'tvn.temporal'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ])
