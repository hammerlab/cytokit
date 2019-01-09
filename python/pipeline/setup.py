from setuptools import setup, find_packages
import os

with open('requirements.txt', 'r') as fd:
    requires = [l.strip() for l in fd.readlines()]

if __name__ == '__main__':
    setup(
        name='cytokit',
        version='0.0.1',
        description="Microscopy Image Cytometry Toolkit",
        author="Eric Czech",
        author_email="eric@hammerlab.org",
        url="",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.6'
        ],
        install_requires=requires,
        extras_require={
            "tf": ["tensorflow>=1.7.0"],
            "tf_gpu": ["tensorflow-gpu>=1.7.0"],
        },
        packages=find_packages(exclude=('tests',)),
        package_data={},
        include_package_data=True,
        entry_points={'console_scripts': ['cytokit = cytokit.cli.main:main']}
    )
