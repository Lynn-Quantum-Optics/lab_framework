from setuptools import setup

setup(
    name='lab_framework',
    packages=['lab_framework'],
    version='1.0.6',
    description='Framework for controlling many components of the lab.',
    url='https://github.com/Lynn-Quantum-Optics/lab_framework',
    author='Alec Roberson',
    author_email='alectroberson@gmail.com',
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'uncertainties', 
        'scipy',
        'pyserial',
        'tqdm',
        'thorlabs_apt'
    ]
)