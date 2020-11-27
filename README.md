Utils for gathering, aggregation and handling metadata from DICOM files.

# Installation

From pip
```
pip install dicom-csv
```

or from GitHub

```bash
git clone https://github.com/neuro-ml/dicom-csv
cd dicom-csv
pip install -e .
```

# Example `join_tree`

```python
>>> from dicom_csv import join_tree
>>> folder = '/path/to/folder/'
>>> meta = join_tree(folder, verbose=2)
>>> meta.head(3)
```
| AccessionNumber | AcquisitionDate |  ...  | WindowCenter | WindowWidth |
| -------------: | -------------:   | :---: | --------:    | :---------: |
|000002621237 	 |20200922          |...    |-500.0        |1500.0       |
|000002621237 	 |20200922          |...    |-40.0         |400.0        |
|000002621237 	 |20200922          |...    |-500.0        |1500.0       |
3 rows x 155 columns



# Documentation

You can find the documentation [here](https://dicom-csv.readthedocs.io/en/latest/index.html).
