# Created by Javad Komijani (2023)

"""This is the setup script for `torch_linalg_ext`."""

from setuptools import setup


def readme():
    """Reads and returns the contents of the README.md file."""
    with open('README.md', encoding='utf-8') as f:
        return f.read()


packages = [
        'torch_linalg_ext',
        'torch_linalg_ext._autograd',
        'torch_linalg_ext._eig',
        'torch_linalg_ext._svd',
        'torch_linalg_ext.functions'
        ]

package_dir = {
        'torch_linalg_ext': 'src',
        'torch_linalg_ext._autograd': 'src/_autograd',
        'torch_linalg_ext._eig': 'src/_eig',
        'torch_linalg_ext._svd': 'src/_svd',
        'torch_linalg_ext.functions': 'src/functions'
        }

setup(name='torch_linalg_ext',
      version='1.0.0',
      description="a package with extensions to torch.linalg.",
      packages=packages,
      package_dir=package_dir,
      url='http://github.com/jkomijani/torch_linalg_ext',
      author='Javad Komijani',
      author_email='jkomijani@gmail.com',
      license='MIT',
      zip_safe=False
      )
