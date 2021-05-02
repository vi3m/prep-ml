from prep_ml.pre_processor import Prep
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

p = Prep.from_dict(prep_ob, df)
rdf = p.get_data()
print(rdf.columns.to_list())
print(rdf.head())