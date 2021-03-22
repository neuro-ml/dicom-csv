from setuptools import setup, find_packages

classifiers = '''Development Status :: 4 - Beta
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
'''

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

# get the current version
with open('dicom_csv/__version__.py', encoding='utf-8') as file:
    scope = {}
    exec(file.read(), scope)
    __version__ = scope['__version__']

setup(
    name='dicom_csv',
    packages=find_packages(include=('dicom_csv',)),
    include_package_data=True,
    version=__version__,
    description='Utils for gathering, aggregation and handling metadata from DICOM files.',
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
            'collect_contours = dicom_csv.scripts:collect_contours',
        ],
    },
)
