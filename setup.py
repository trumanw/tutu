import sys
import time
from setuptools import setup, find_packages

# Environment-specific dependencies.
extras = {
    'torch': ['torch', 'torchvision', 'dgl', 'dgllife'],
}

setup(
    name='tutu',
    verison='0.1.0',
    url='https://github.com/trumanw/tutu',
    aintainer='TuTu contributors',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    license='MIT',
    description='Deep learning models for material science.',
    keywords=[
        'tutu',
        'chemistry',
        'surfactant',
        'materials-science',
    ],
    packages=find_packages(exclude=["*.tests"]),
    install_requires=[
        'streamlit',
        'numpy>=1.21',
        'pandas',
        'scikit-learn',
        'rdkit-pypi',
        'seaborn',
        'matplotlib',
        'plotly',
    ],
    python_requires='>=3.7,<3.10')