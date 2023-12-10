import pandas as pd
from sklearn.model_selection import train_test_split

class DataCleaning:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.data_splitter = DataSplitter()

    def process_and_split_data(self, df: pd.DataFrame):
        """Preprocesses and divides the data"""
        processed_data = self.data_processor.process_data(df = df)
        X_train, X_test, y_train, y_test = self.data_splitter.split_data(df = processed_data)
        return X_train, X_test, y_train, y_test

class DataProcessor:
    """ Preprocesses the data"""
    def process_data(self, df: pd.DataFrame):
        self.drop_useless_cols(df = df)
        self.find_and_handle_outliers(columns = ['Impressions', 'Clicks', 'Spent', 'Total_Conversion', 'Approved_Conversion'])
        self.create_age_bounds(df = df)
        return df

    def drop_useless_cols(self, df: pd.DataFrame):
        not_required_cols = ['ad_id', 'xyz_campaign_id', 'fb_campaign_id']
        df.drop(columns=not_required_cols, axis = 1, inplace= True)
    
    def find_and_handle_outliers(self, columns: pd.Series):
        """Uses iqr method to get minium and maxium threshold for a value to be outlier"""
        for col in columns:
            # Calculate lower and upper bound with IQR
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            # Calculate total outliers and append if has > 0 outiers
            total_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if total_outliers > 0:
                   df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    def create_age_bounds(self, df: pd.DataFrame):
        """ Create new columns seperating min and max values of a bin"""
        df['age_lower_bound'] = df['age'].apply(lambda age: age[0:2])
        df['age_upper_bound'] = df['age'].apply(lambda age: age[3:5])

class DataSplitter:
    """ Splits the data into train and test"""
    def split_data(self, df: pd.DataFrame):
        X = df.drop(columns=['Approved_Conversion'], axis=1)
        y = df[['Approved_Conversion']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


df = pd.read_csv('/home/diwas/Documents/DevStuff/Sales-Conversion-Optimization-Project/data/raw/KAG_conversion_data.csv')
data_cleaning = DataCleaning()
X_train, X_test, y_train, y_test = data_cleaning.process_and_split_data(df)