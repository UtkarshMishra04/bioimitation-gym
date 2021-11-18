from setuptools import setup
from setuptools import find_packages

with open('./scripts/requirements.txt') as f:
    required = f.read().splitlines()

required.append('opensim==4.1')

setup(
    name='bioimitation',
    packages=find_packages(),
    package_data={'bioimitation.imitation_envs': [
        'data/*',
    ]},
    version='0.1',
    description='Biomechanical Human Gait Imitation',
    long_description=open('./README.md').read(),
    author='Utkarsh A. Mishra, Dimitar Stanev',
    author_email='utkarsh75477@gmail.com',
    url='https://utkarshmishra04.github.io/bioimitation-gym',
    requires=(),
    install_requires=required,
    zip_safe=True,
    license='MIT',
    python_requires=">=3.6"
)