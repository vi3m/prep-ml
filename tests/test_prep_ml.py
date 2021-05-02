from prep_ml import __version__
from prep_ml.pre_processor import Prep
import pytest
from prep_ml.exceptions import DataMissingError, IncorrectSchemaError, JSONDeserializationError
import pandas as pd


prep_ob = {
    "EMPLOYEE_ID": {
        "alias": "EMP_ID",
        "required": True
    },
    "FIRST_NAME": {
        "required": False,
    },
    "LAST_NAME": {
        "required": False,
    },
    "AGE": {
        "required": True,
        "alias": "DOB",
        "derived_eq": "(pd.Timestamp('now') - pd.to_datetime(df.AGE, format='%m/%d/%Y')).astype('<m8[Y]')"
    },
    "GENDER": {
        "required": True,
        "encoding": 'ohe'
    },
    "RATING": {
        "required": True,
        "imputation": 'most_frequent'
    },
    "DESIGNATION": {
        "required": True,
        "encoding": 'label'
    },
    "PROMOTED": {
        "required": True,
        "encoding": 'label'
    }
}
df = pd.read_csv('tests/MOCK_DATA.csv')


def test_data_missing_exception():
    with pytest.raises(DataMissingError) as e:
        Prep.from_dict(prep_ob)
        assert 'Missing file_path and/or dataframe.' in str(e.value)


def test_json_error():
    json_ob = '''{
        "EMPLOYEE_ID": {
            "required": True
        }
    }'''
    with pytest.raises(JSONDeserializationError) as e:
        Prep.from_json(json_ob, df)
        assert 'Error parsing json' in str(e.value)


def test_mapper_schema_failed():
    error_ob = {
        "EMPLOYEE_ID": "FAIL_VALUE"
    }
    Prep.from_dict(prep_ob, df)
    with pytest.raises(IncorrectSchemaError) as e:
        Prep.from_dict(error_ob, df)
        assert 'Error with mapper schema.' in str(e.value)


def test_mapper_schema_new_key():
    error_ob = {
        "EMPLOYEE_ID": [{
            "required": True,
            "encoding": None,
            "alias": "EMP_ID",
            "imputation": None,
            "enforce_type": None,
            "derived_eq": None,
            "some_new_key": None
        }]
    }
    Prep.from_dict(prep_ob, df)
    with pytest.raises(IncorrectSchemaError) as e:
        Prep.from_dict(error_ob, df)
        assert 'Unknown keys found in schema.' in str(e.value)
    

def test_alias_features():
    p = Prep.from_dict(prep_ob, df)
    p.get_data()
    assert p.get_alias_features() == ['EMPLOYEE_ID', 'AGE']

def test_required_features():
    p = Prep.from_dict(prep_ob, df)
    p.get_data()
    assert p.get_required_features() == ['EMPLOYEE_ID','AGE','GENDER','RATING','DESIGNATION','PROMOTED']

def test_encode_features():
    p = Prep.from_dict(prep_ob, df)
    p.get_data()
    assert p.get_encode_features() == ['GENDER', 'DESIGNATION', 'PROMOTED']

def test_imputation_features():
    p = Prep.from_dict(prep_ob, df)
    p.get_data()
    assert p.get_imputation_features() == ['RATING']

def test_derived_features():
    p = Prep.from_dict(prep_ob, df)
    p.get_data()
    assert p.get_derived_features() == ['DOB']

def test_derived_features():
    p = Prep.from_dict(prep_ob, df)
    rdf = p.get_data()
    assert not all(item in rdf.columns.tolist() for item in ['FIRST_NAME', 'LAST_NAME'])

def test_imputation_process():
    p = Prep.from_dict(prep_ob, df)
    rdf = p.get_data()
    assert not rdf.isnull().values.any()