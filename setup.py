from setuptools import setup, find_packages

from dicom_csv import __version__

classifiers = '''Development Status :: 4 - Beta
Programming Language :: Python :: 3.6'''

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

setup(
    name='dicom_csv',
    packages=find_packages(include=('dicom_csv',)),
    include_package_data=True,
    version=__version__,
    descriprion='Utils for gathering, aggregation and handling metadata from DICOM files.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/neuro-ml/dicom-csv',
    download_url='https://github.com/neuro-ml/dicom-csv/v%s.tar.gz' % __version__,
    keywords=[],
    classifiers=classifiers.splitlines(),
    install_requires=requirements,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'dicom-csv = dicom_csv.scripts:join_to_csv',
        ],
    },
)
