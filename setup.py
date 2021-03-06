from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()


version = '0.1'

install_requires = [
    # List your project dependencies here.
    # For more details, see:
    # http://packages.python.org/distribute/setuptools.html#declaring-dependencies
    'numpy',
    'scipy',
]


setup(name='PSIFA',
    version=version,
    description="Python implementation of the Pattern Search Implicit Filtering Algorithm (PSIFA)",
    long_description=README + '\n\n' + NEWS,
    classifiers=[
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3.6',
      'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='derivative-free-optimization linearly-constrained-minimization noisy-optimization global-convergence degenerate-constraints',
    author='Carlos H. Villa Pinto',
    author_email='chvillap@gmail.com',
    url='https://github.com/chvillap',
    license='GNU GPLv3',
    packages=find_packages('src'),
    package_dir = {'': 'src'},
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        'console_scripts':
            ['PSIFA=psifa:main']
    }
)
