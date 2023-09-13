import os
import pandas as pd
import numpy as np
import env

def acquire_zillow():
    '''
    acquire_zillow will use a local env.py
    using pre-set credentials called user, password, and host
    please make sure you have a properly formatted env.py
    file in the same directory as this module
    and that you have the access rights to zillow schema
    
    return: a single pandas dataframe
    '''
    if os.path.exists('zillow_data.csv'):
        df = pd.read_csv('zillow_data.csv')
    else:
        query = '''SELECT *
                FROM properties_2017
                LEFT JOIN predictions_2017 ON properties_2017.id = predictions_2017.id
                LEFT JOIN architecturalstyletype ON properties_2017.architecturalstyletypeid = architecturalstyletype.architecturalstyletypeid
                LEFT JOIN buildingclasstype ON properties_2017.buildingclasstypeid = buildingclasstype.buildingclasstypeid
                LEFT JOIN heatingorsystemtype ON properties_2017.heatingorsystemtypeid = heatingorsystemtype.heatingorsystemtypeid
                LEFT JOIN storytype ON properties_2017.storytypeid = storytype.storytypeid
                LEFT JOIN propertylandusetype ON properties_2017.propertylandusetypeid = propertylandusetype.propertylandusetypeid
                LEFT JOIN typeconstructiontype ON properties_2017.typeconstructiontypeid = typeconstructiontype.typeconstructiontypeid
                LEFT JOIN airconditioningtype ON properties_2017.airconditioningtypeid = airconditioningtype.airconditioningtypeid
                LEFT JOIN unique_properties ON properties_2017.parcelid = unique_properties.parcelid
                WHERE properties_2017.propertylandusetypeid = 261
                AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL'''
        url = env.create_url('zillow')
        df = pd.read_sql(query, url)
        df.to_csv('zillow_data.csv', index=False)
    df = df.drop(columns=[ 
                    'airconditioningtypeid',
                    'architecturalstyletypeid', 
                    'buildingclasstypeid',
                    'buildingqualitytypeid',
                    'decktypeid', 
                    'pooltypeid10',
                    'pooltypeid2',
                    'pooltypeid7', 
                    'propertylandusetypeid',
                    'storytypeid',
                    'typeconstructiontypeid', 
                    'id.1',
                    'parcelid.1',
                    'architecturalstyletypeid.1',
                    'buildingclasstypeid.1',
                    'heatingorsystemtypeid.1', 
                    'storytypeid.1',
                    'propertylandusetypeid.1', 
                    'typeconstructiontypeid.1',
                    'airconditioningtypeid.1',
                    'parcelid.2',
                    'heatingorsystemtypeid'])
    return df


def missing_by_col(df): 
    '''
    returns a single series of null values by column name
    '''
    # get count of missing elements by column (axis 0)
    count_rows_missing = df.isnull().sum(axis=0)
    # get the ratio of missing elements by row:
    percent_rows_missing = round((df.isnull().sum(axis=0) / len(df)) * 100)
    
    # make df with those two series
    cols_df = pd.DataFrame({
    'num_rows_missing': count_rows_missing,
    'percent_rows_missing': percent_rows_missing
    }).reset_index().groupby(['num_rows_missing', 'percent_rows_missing']).count().reset_index().rename(columns={'index': 'num_cols'})
    
    return cols_df

def missing_by_row(df):
    '''
    prints out a report of how many rows have a certain
    number of columns/fields missing both by count and proportion
    
    '''
    # get the number of missing elements by row (axis 1)
    count_missing = df.isnull().sum(axis=1)
    # get the ratio/percent of missing elements by row:
    percent_missing = round((df.isnull().sum(axis=1) / df.shape[1]) * 100)
    # make a df with those two series (same len as the original df)
    # reset the index because we want to count both things
    # under aggregation (because they will always be sononomous)
    # use a count function to grab the similar rows
    # print that dataframe as a report
    rows_df = pd.DataFrame({
    'num_cols_missing': count_missing,
    'percent_cols_missing': percent_missing
    }).reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).\
    count().reset_index().rename(columns={'index':'num_rows'})
    return rows_df

def get_fences(df, col, k=1.5) -> tuple:
    '''
    get fences will calculate the upper and lower fence
    based on the inner quartile range of a single Series
    
    return: lower_bound and upper_bound, two floats
    '''
    q3 = df[col].quantile(0.75)
    q1 = df[col].quantile(0.25)
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    lower_bound = q1 - (k * iqr)
    return lower_bound, upper_bound

def report_outliers(df, k=1.5) -> None:
    '''
    report_outliers will print a subset of each continuous
    series in a dataframe (based on numeric quality and n>20)
    and will print out results of this analysis with the fences
    in places
    '''
    num_df = df.select_dtypes('number')
    for col in num_df:
        if len(num_df[col].value_counts()) > 20:
            lower_bound, upper_bound = get_fences(df,col, k=k)
            print(f'Outliers for Col {col}:')
            print('lower: ', lower_bound, 'upper: ', upper_bound)
            print(df[col][(
                df[col] > upper_bound) | (df[col] < lower_bound)])
            print('----------')


def summarize(df, k=1.5) -> None:
    '''
    Summarize will take in a pandas DataFrame
    and print summary statistics:
    
    info
    shape
    outliers
    description
    missing data stats
    
    return: None (prints to console)
    '''
    # print info on the df
    print('Shape of Data: ')
    print(df.shape)
    print('======================\n======================')
    print('Info: ')
    print(df.info())
    print('======================\n======================')
    print('Descriptions:')
    # print the description of the df, transpose, output markdown
    print(df.describe().T.to_markdown())
    print('======================\n======================')
    # lets do that for categorical info as well
    # we will use select_dtypes to look at just Objects
    print(df.select_dtypes('O').describe().T.to_markdown())
    print('======================\n======================')
    print('missing values:')
    print('by column:')
    print(missing_by_col(df).to_markdown())
    print('by row: ')
    print(missing_by_row(df).to_markdown())
    print('======================\n======================')
    print('Outliers: ')
    print(report_outliers(df, k=k))
    print('======================\n======================')
    
def drop_multi_fam(df):
    '''
    function takes in dataframe and returns dataframe where unitcnt is not 2, 3, or 4
    '''
    df = df[(df['unitcnt'] != 2) & (df['unitcnt'] != 3) & (df['unitcnt'] != 4)]
    return df
    
def handle_missing_values(df, prop_required_column, prop_required_row):
    '''
    function takes in 3 positional arguments (dataframe, proportion of complete rows in column required, and proportion of columns in row required)
    
    returns: df with the columns and rows that have not met our conditions of prop_required_column and prop_required_row
    '''
    # loop to drop columns that do not meet prop_required_column
    for column in df.columns:
        if (df[column].isnull().sum() / len(df)) > (1 - prop_required_column):
            df = df.drop(columns=[column])

    # loop to drop rows that do not meet prop_required_row
    for row in df.index:
        if (df.loc[row].isnull().sum() / df.shape[1]) > (1 - prop_required_row):
            df = df.drop(index=row)

    return df
