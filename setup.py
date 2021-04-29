from setuptools import setup, find_packages

NAME = 'torchmdnet2'
VERSION = '0.1'

install_requires = [
    'mdtraj',
    'jsonargparse[signatures]',
    'tqdm',
    'ase',
    'e3nn',
    'numpy',
    'scipy'
]


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    zip_safe=True,
    python_requires='>=3.8',
    install_requires=install_requires,
    )