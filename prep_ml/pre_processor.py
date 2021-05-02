import numpy as np
import pandas as pd
import json
from .exceptions import DataMissingError, IncorrectSchemaError, TypeMismatchError, JSONDeserializationError, UnsupportedEncodingError
from sklearn.impute import SimpleImputer
import category_encoders as ce

class Prep:
    def __init__(self, mapper, df) -> None:
        self.mapper = mapper
        self.df = df

        self.required = None
        self.encoding = None
        self.alias = None
        self.imputation = None
        self.derived = None

        self.rename_mapper = {}

        self.extract_ob = {
            "required": [],
            "encoding": [],
            "alias": [],
            "imputation": [],
            "derived_eq": []
        }

        is_valid, ops = self.validate_schema(mapper)
        
        if not is_valid:
            raise IncorrectSchemaError('Unknown keys found in schema.')

        self.process()


    def process(self):
        self.extract_mapper()
        self.rename_mapper_constructor()
        self.process_alias()
        self.process_required()
        self.process_impute()
        self.process_derivation()
        self.process_encoding()
    

    def get_data(self):
        '''
        Returns the processed dataframe.
        '''
        return self.df


    def rename_mapper_constructor(self):
        for item in self.extract_ob['alias']:
            self.rename_mapper[self.mapper.get(item).get('alias')] = item


    def process_alias(self):
        self.df.rename(columns=self.rename_mapper, inplace=True)


    def process_required(self):
        self.df = self.df[self.extract_ob['required']]
    
    def process_impute(self):
        for feature in self.extract_ob['imputation']:
            strategy = self.mapper.get(feature).get('imputation')
            imputer = SimpleImputer(missing_values = np.nan, strategy = strategy)
            imp_arr = imputer.fit_transform(np.array(self.df[feature]).reshape(-1,1))
            self.df = self.df.drop(columns=[feature])
            self.df = pd.concat([self.df, pd.DataFrame(imp_arr, columns=[feature])], axis=1)
    
    def process_derivation(self):
        for feature in self.extract_ob['derived_eq']:
            eqn = self.mapper.get(feature).get('derived_eq')
            drv_df = eval(eqn.replace('df', 'self.df'))
            self.df = self.df.drop(columns=[feature])
            self.df = pd.concat([self.df, drv_df], axis=1)
    

    def process_encoding(self):
        for feature in self.extract_ob['encoding']:
            strategy = self.mapper.get(feature).get('encoding')
            if strategy == 'label':
                encoder = ce.OrdinalEncoder(cols=[feature])
                self.df[feature] = encoder.fit_transform(self.df[feature])
            elif strategy == 'ohe':
                encoder = ce.OneHotEncoder(cols=[feature])
                enc_df = encoder.fit_transform(self.df[feature])
                self.df = self.df.drop(columns=[feature])
                self.df = pd.concat([self.df, enc_df], axis=1)
            else:
                raise UnsupportedEncodingError()

    
    def get_mapper(self):
        return self.mapper
    
    def get_required_features(self):
        return self.extract_ob.get('required')
    
    def get_encode_features(self):
        return self.extract_ob.get('encoding')
    
    def get_alias_features(self):
        return self.extract_ob.get('alias')
    
    def get_imputation_features(self):
        return self.extract_ob.get('imputation')
    
    def get_derived_features(self):
        return self.extract_ob.get('derived_eq')

    
    def extract_mapper(self):
        for feature, schema in self.mapper.items():
            for k,v in schema.items():
                self.extract_ob[k].append(feature) if v else ...



    @classmethod
    def from_json(cls, mapper, df=None):
        '''
        Reads the mapper JSON string object and performs operations on provided dataframe as defined in the mapper.
        
        Args:
            mapper: A JSON mapper object.
            df: pandas DataFrame.
        '''
        if df is None:
            raise DataMissingError('Missing file_path and/or dataframe.')
        
        if not isinstance(df, pd.DataFrame):
            raise TypeMismatchError('Not a DataFrame.')
        
        if not isinstance(mapper, str):
            raise TypeMismatchError('Not a JSON String.')
        
        try:
            mapper_dict = json.loads(mapper)
        except Exception as e:
            raise JSONDeserializationError('Error parsing json. Ensure mapper is JSON compliant.')
        
        return cls(mapper_dict, df)


    @classmethod
    def from_dict(cls, mapper, df=None):
        '''
        Reads the mapper python dict object and performs operations on provided dataframe as defined in the mapper.
        
        Args:
            mapper: A python dict mapper object.
            df: pandas DataFrame.
        '''
        if df is None:
            raise DataMissingError('Missing file_path and/or dataframe.')
        
        if not isinstance(df, pd.DataFrame):
            raise TypeMismatchError('Not a DataFrame.')
        
        if not isinstance(mapper, dict):
            raise TypeMismatchError('Not a python dictonary')
        
        return cls(mapper, df)
    

    def validate_schema(self, mapper):
        keys = set()
        for v in mapper.values():
            try:
                keys.update(v.keys())
            except Exception:
                raise IncorrectSchemaError('Error with the mapper schema.')
        permitted_keys = {'alias', 'required', 'encoding', 'imputation', 'derived_eq'}
        return keys.issubset(permitted_keys) or keys == permitted_keys, keys