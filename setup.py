# Copyright (c) 2022 Javad Komijani


from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


packages = [
        'torch_linalg_ext',
        'torch_linalg_ext._eig'
        ]

package_dir = {
        'torch_linalg_ext': 'src',
        'torch_linalg_ext._eig': 'src/_eig'
        }

setup(name='torch_linalg_ext',
      version='0.0',
      description="a package with extensions to torch.linalg.",
      packages=packages,
      package_dir=package_dir,
      url='http://github.com/jkomijani/torch_linalg_ext',
      author='Javad Komijani',
      author_email='jkomijani@gmail.com',
      license='MIT',
      install_requires=['numpy>=1.23'],
      zip_safe=False
      )
