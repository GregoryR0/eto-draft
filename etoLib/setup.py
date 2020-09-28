from setuptools import setup

setup(name='etoLib',
      maintainer='Greg Rouze',
      maintainer_email='rouze@contractor.usgs.gov',
      version='1.0.0',
      description='Classes and Functions for et reference data creation input to water balance',
      packages=[
          'etoLib',
      ],
      install_requires=[
          'boto3',
      ],

)
