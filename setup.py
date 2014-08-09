from setuptools import setup, find_packages

setup(
    name = 'statisty',
    version = '0.1',
    classifiers = [
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'License :: OSI Approved :: GPL',
        'Topic :: Statistics'
    ],
    package_data = {
        '': ['*.rst']
    },
    install_requires = ['numpy>=1.7.1', 'scipy>=0.13', 'future>=0.12'],
    license = 'GPL 3',
    author = 'Seonghyeon Kim',
    author_email = 'kaleana@gmail.com',
    description = 'Python statistical tools for social sciences.',
    packages = find_packages()
)
