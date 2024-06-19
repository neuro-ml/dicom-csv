from pathlib import Path

from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3 :: Only',
]
name = 'dicom_csv'
root = Path(__file__).resolve().parent
with open(root / 'README.md', encoding='utf-8') as file:
    long_description = file.read()

with open(root / 'requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

# get the current version
with open(root / name / '__version__.py', encoding='utf-8') as file:
    scope = {}
    exec(file.read(), scope)
    __version__ = scope['__version__']

setup(
    name=name,
    packages=find_packages(include=(name,)),
    include_package_data=True,
    version=__version__,
    description='Utils for gathering, aggregation and handling metadata from DICOM files.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/neuro-ml/dicom-csv',
    download_url='https://github.com/neuro-ml/dicom-csv/v%s.tar.gz' % __version__,
    keywords=['DICOM'],
    classifiers=classifiers,
    install_requires=requirements,
    extras_require={
        'nifti': 'nibabel',
        'all': 'nibabel',
    },
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'dicom-csv = dicom_csv.scripts:join_to_csv',
            'collect_contours = dicom_csv.scripts:collect_contours',
        ],
    },
)
