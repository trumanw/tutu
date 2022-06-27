import sys
import time
from setuptools import setup, find_packages

if '--release' in sys.argv:
    IS_RELEASE = True
    sys.argv.remove('--release')
else:
    # Build a nightly package by default.
    IS_RELEASE = False

# Environment-specific dependencies.
extras = {
    'torch': ['torch', 'torchvision', 'dgl', 'dgllife'],
}

# get the version from tutu/__init__.py
def _get_version():
    with open('tutu/__init__.py') as fp:
        for line in fp:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)
                base = g['__verison__']
                if IS_RELEASE:
                    return base
                else:
                    # nightly version : .devYearMonthDayHourMinute
                    if base.endswith('.dev') is False:
                        # Force to add `.dev` if `--release` option isn't passed when building
                        base += '.dev'
                    return base + time.strftime("%Y%m%d%H%M%S")
    raise ValueError('`__version__` not defined in `tutu/__init__.py`')


setup(
    name='tutu',
    verison=_get_version(),
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
        'scipy',
        'rdkit-pypi',
        'seaborn',
        'matplotlib',
        'plotly',
        'rdkit-pypi',
    ],
    python_requires='>=3.7,<3.10')