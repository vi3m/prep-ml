<h1 align="center">Prep-ML</h1>

[![Downloads](https://static.pepy.tech/personalized-badge/prep-ml?period=month&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/prep-ml)

## What is Prep-ML?
prep-ml is an open-source pre-processing library aimed at simplifying the data processing steps and streamlining the transformation techniques before feeding it to your choice of machine learning algorithm.

<br>

## Why Prep-ML?
Production grade machine learning is quite different from the standard notebook building. Notebook building is aimed at fast development, interactive code, and visual feedback system. While the scripts aim to cater models to large groups of audience or companies.

For eg, consider one of the key features of your model is DATE_OF_BIRTH, in real-time, due to various database schemas, the feature could be available in any of its synonyms (say, DOB, BIRTH_DATE). This is where prep-ml tries to fill in, like a heavily inspired from ETL tools and other design patterns.

<br>

## Installation

``` $ pip install prep-ml```

<br>

## Documentation

This is the schema currently supported by the library. This can take python dict or JSON string.
```
{
    "FEATURE_NAME": {
        "required": bool,
        "encoding": str,
        "alias": str,
        "imputation": str,
        "derived_eq": str
    }
}
```
### Schema Definitions:

#### required: bool

> accepted values: **True**, **False**

determines if the feature is required for the model. 
- If **required** is set to **False**, the **FEATURE_NAME** is discarded for further processing.

#### encoding: str

> accepted values: **label**, **ohe**

performs the given encoding strategy on the **FEATURE_NAME**. 

- If **encoding** is set to "**label**", LabelEncoding or OrdinalEncoding is performed on the **FEATURE_NAME**
- If **encoding** is set to "**ohe**", OneHotEncoding is performed on the **FEATURE_NAME**


#### alias: str

> accepted values: any string

this is a synonym or alias for the given **FEATURE_NAME**. 

- For eg, If **alias** is set to **"FEATURE_OTHER_NAME"**, the alias name will be mapped to the **FEATURE_NAME**


#### imputation: str

> accepted values: **mean**, **median**, **most_frequent**

performs the given imputation strategy on the **FEATURE_NAME**. This is a wrapper of SimpleImputer. 

- If "**mean**", then replace missing values using the mean for the **FEATURE_NAME**. Can only be used with numeric data.
- If "**median**", then replace missing values using the median for the **FEATURE_NAME**. Can only be used with numeric data.
- If "**most_frequent**", then replace missing using the most frequent value for the **FEATURE_NAME**. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.


#### derived_eq: str

> accepted values: eval equation as a string

evaluated the given equation and then assigns the response to **FEATURE_NAME**. The reference to dataframe should be **df**

- For eg consider the above feature *DOB*, If **derived_eq** is set to "**pd.to_datetime(df.DOB, format='%m/%d/%Y')**", the expression will be evaluated and assigned to **FEATURE_NAME**. Note that, **df** is reference to the provided input df.



### Methods:

> **from_dict(mapper, dataframe)** -- reads the dict and processes the input dataframe.

> **from_json(json_mapper, dataframe)** -- reads the json and processes the input dataframe.

> **get_data()** --  fetches the processed data.



<br>


### Usage Example:
This is the input data 

![input_data](https://github.com/vi3m/vi3m_image_host/blob/master/prep-ml/readme_input.png?raw=true)

#### Data Explanation:

This is randomly generated data for the purposes of demo. All references are assumptions.

This is a company employee data. We have various features, which are self explanatory.

Ideally, we would want to remove the NAMES, as they are uniques and serve no purpose in model. Transform DOB to say a derived feature called AGE. Encode, GENDER, DESIGNATION and PROMOTED. Impute RATING.

So, on using the driver code.

```
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

```
Output
```
['EMPLOYEE_ID', 'DESIGNATION', 'PROMOTED', 'RATING', 'AGE', 'GENDER_1', 'GENDER_2', 'GENDER_3']
```

The output in dataviewer is as follows.
![readme_processed](https://github.com/vi3m/vi3m_image_host/blob/master/prep-ml/readme_processed.png?raw=true)


<br>

## Future Development Roadmap
- Performance improvements.
- Add support for more imputation and encoding strategies.
- Support for feature scaling.
- Support for multiple schemas.
- Support for multiple input sources.
- Support for enforcing column types.
- Feasibility for model training.

<br>

## Changelog

2nd May, 2021 :: v0.1.0:
- This is a very early dev version. This further needs development and code optimization.