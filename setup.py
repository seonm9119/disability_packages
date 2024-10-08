from setuptools import setup, find_packages

setup(name='disability',
      version='0.0.0',
      description='Communication Project for People with Disabilities',
      url='https://github.com/seonm9119/disability_packages.git',
      author='nami',
      author_email='seonm9119@gmail.com',
      packages=find_packages(include=['disability', 'organizer']),
      install_requires=['pandas', 'moviepy', 'librosa'])