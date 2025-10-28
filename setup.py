from setuptools import find_packages
from distutils.core import setup

setup(name='unitree_rl_mugym',
      version='1.0.0',
      author='Unitree Robotics (Modified for Mujoco/Colab)',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='support@unitree.com',
      description='Mujoco-based RL environments for Unitree Robots - Train on Colab, visualize locally',
      url='https://github.com/julienokumu/unitree_rl_mugym',
      install_requires=['isaacgym', 'rsl-rl-lib', 'matplotlib', 'numpy>=1.20', 'tensorboard', 'mujoco==3.2.3', 'pyyaml', 'scipy'],
      extras_require={
          'mujoco_only': ['rsl-rl-lib', 'matplotlib', 'numpy>=1.20', 'tensorboard', 'mujoco==3.2.3', 'pyyaml', 'scipy', 'torch']
      })
