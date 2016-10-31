#!/usr/bin/env python

from setuptools import setup

numpy_version = 'numpy>=1.11.2'
scipy_version = 'scipy>=0.18.0'
pandas_version = 'pandas'
six_version = 'six>=1.10.0'
chainer_version = 'chainer>=1.17.0'
pythonjsonlogger_version = 'python-json-logger'
progressbar_version = 'progressbar2'
install_requires = [ 
        numpy_version,
        scipy_version, 
        pandas_version, 
        six_version, 
        chainer_version, 
        progressbar_version,
        pythonjsonlogger_version ]

setup(
        name = 'masalachai',
        version = '0.5.4',
        packages = ['masalachai',
                    'masalachai.datafeeders',
                    'masalachai.preprocesses',
                    'masalachai.posttest_processes',
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
