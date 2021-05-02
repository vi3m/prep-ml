class PrepException(Exception):
    '''
    Base Exception with all PerpML related Exceptions.
    '''
    
    ...


class DataMissingError(PrepException):
    '''
    Missing either file_path or dataframe.
    '''
    
    ...


class TypeMismatchError(PrepException):
    '''
    Expected type does not match the provided input type.
    '''
    
    ...


class JSONDeserializationError(PrepException):
    '''
    Error while deserializing JSON.
    '''
    
    ...


class IncorrectSchemaError(PrepException):
    '''
    Error with mapper schema.
    '''
    
    ...

class UnsupportedEncodingError(PrepException):
    '''
    Unsupported encoder.
    '''
    
    ...