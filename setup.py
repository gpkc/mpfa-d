#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


requirements = ['pymoab', ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', 'pytest-cov', 'pytest-mock']

setup(
    author="PADMEC",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='mpfad',
    name='mpfad',
    packages=find_packages(include=['mpfad']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    version='0.0.1',
    zip_safe=False,
)
