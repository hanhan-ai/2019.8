#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='HanhanAI',
    version='1.0.0',
    description=
        'Universal game AI system for game developer',
    long_description=open('README.rst').read(),
    author='HanhanAI producer1',
    author_email='1379612504@qq.com',
    maintainer='HanhanAI producer2',
    maintainer_email='762613908@qq.com',
    license='MIT',
    packages=['HanhanAI'],
    platforms=["all"],
    url='https://github.com/hanhan-ai/2019HanHan',#github网址
    install_requires=[ 'keras','numpy','matplotlib'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
)
