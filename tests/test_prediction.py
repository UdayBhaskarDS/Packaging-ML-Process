import pytest

from pathlib import Path
import os
import sys

# Adding the below path to avoid module not found error
# Set the path to the project root
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PACKAGE_ROOT))


from predictive_modelling.config import config
from predictive_modelling.processing.data_handling import load_dataset
from predictive_modelling.predict import generate_predictions

# output from predict script not null
# output from predict script is str data type
# the output is Y for an example data

#Fixtures --> functions before test function --> ensure single_prediction

@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result

def test_single_pred_not_none(single_prediction): # output is not none
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction): # data type is string
    assert isinstance(single_prediction.get('prediction')[0],str)

def test_single_pred_validate(single_prediction): # check the output is Y
    assert single_prediction.get('prediction')[0] == 'Y'