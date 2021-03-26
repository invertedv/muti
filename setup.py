from distutils.core import setup

setup(
    name='utilities',
    version='1.0',
    packages=['utilities'],
    url='',
    license='MIT',
    author='William Alexander',
    author_email='will@invertedv.com',
    install_requires=['numpy', 'pandas', 'tensorflow', 'plotly',
                      'modeling @ git+https://github.com/invertedv/modeling'],
description='Tools for regression modeling'
)
