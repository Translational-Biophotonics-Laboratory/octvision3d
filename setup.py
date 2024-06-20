from setuptools import setup, find_packages

setup(
    name='octvision3d',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['bin/*.sh'],
    },
    entry_points={
        'console_scripts': [
            'train_nnUNet_2d=bin.train_nnUNet_2d:main',
            'train_nnUNet_3d=bin.train_nnUNet_3d:main',
            'train_nnUNet_3dcascade=bin.train_nnUNet_3dcascade:main',
        ]'
    }
)
