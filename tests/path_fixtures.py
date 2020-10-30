from pathlib import Path

import pytest


@pytest.fixture
def tests_folder():
    return Path('~/dicom-csv-test/').expanduser()
