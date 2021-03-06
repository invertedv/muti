from distutils.core import setup

setup(
    name='muti',
    version='1.0',
    packages=['muti'],
    url='',
    license='MIT',
    author='William Alexander',
    author_email='will@invertedv.com',
    install_requires=['numpy', 'pandas', 'tensorflow', 'plotly', 'scipy', 'kaleido',
                      'modeling'],
description='Tools for regression modeling'
)
