#!/usr/bin/env python

from setuptools import setup

pandas_version = 'pandas'
scipy_version = 'scipy>=0.18.0'
six_version = 'six>=1.10.0'
chainer_version = 'chainer>=1.17.0'
pythonjsonlogger_version = 'python-json-logger'
progressbar_version = 'progressbar2'
install_requires = [ 
        pandas_version, 
        scipy_version, 
        six_version, 
        chainer_version, 
        progressbar_version,
        pythonjsonlogger_version ]

setup(
        name = 'masalachai',
        version = '0.5.2',
        packages = ['masalachai',
                    'masalachai.datafeeders',
                    'masalachai.preprocesses',
                    'masalachai.trainers',
                    'masalachai.loggers',
                    'masalachai.models',
                    'masalachai.optimizer_schedulers'],
        description = 'Masala Chai: easy chainer tool',
        author = 'Daiki Shimada',
        author_email = 'daiki.shimada.9g@stu.hosei.ac.jp',
        url = 'https://github.com/DaikiShimada/masalachai.git',
        install_requires = install_requires,
        license = 'http://www.apache.org/licenses/LICENSE-2.0'
    )
