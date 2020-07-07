Loading contours stored in DICOM format (RTstructure).
======================================================

Segmentation mask is stored in DICOMs in as a set of plain contours. These contours are
nothing but a 2D curves defined by set of coordinates in 3D space. Specifically these coordinates
are in physical space not in voxel space. Every single RTStructure might contain contours for multiple
masks (e.g. `Brain`, `Left_Eye`, `GTV` etc.).

``dicom-csv`` contains several functions to collect contours
stored in RTstructures and move them into image's voxel space.

Crawling images folder
----------------------

``join_tree`` is the main function that collects the DICOM files'
metadata:

.. code-block:: python3

    from dicom_csv import join_tree

    df = join_tree(path, relative=False, verbose=False)

Select rows related to RTStructures
-------------------------------------


``collect_rtstruct`` is the function which selects rows related to RTstructures
and copies additional metadata from corresponding images (RTstructure itself does not contain information about
spatial orientation, it only contains coordinates and link to corresponding DICOM image):

.. code-block:: python3

    from dicom_csv.rtstructs.csv import collect_rtstructs

    df_rtstructs = collect_rtstructs(df)

Extract contours coordinates from RTStructure
----------------------------------------------
``contours_dict`` extract all coordinates from a single row of ``df_rtstructs``:

.. code-block:: python3

    from dicom_csv.rtstruct.contour import read_rtstruct

    contours_dict = read_rtstruct(df_rtstructs.iloc[0])

It outputs python dictionary with keys - names of the contours and values - ``n x 3`` nd.arrays of coordinates
in physical space.

Move contours to voxel space
-----------------------------

Finally, ``contours_image_dict`` moves these coordinates into voxel space (basis change transformation,
essentially rotation, translation and stretching):

.. code-block:: python3

    from dicom_csv.rtstruct.contour import contours_image_dict

    contours_image_dict = contours_to_image(df_rtstructs.iloc[0], contours_dict)

Resulting dictionary contains all contours stored in corresponding RTStructure.

Contours to masks
-----------------

TODO
