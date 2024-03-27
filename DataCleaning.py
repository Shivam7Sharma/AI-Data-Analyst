import pandas as pd
import re
from dateutil import parser


def handle_missing_values(df):
    # Fill missing values with appropriate value
    # This is just an example, you may need to handle missing values differently based on your data
    df.fillna(0, inplace=True)
    return df


def remove_duplicates(df):
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    return df


# def convert_datatypes(df):
#     # Convert data types of columns if necessary
#     # This is just an example, you may need to convert data types differently based on your data
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             df[col] = df[col].astype('category')
#     return df


def clean_data(df):
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    # df = convert_datatypes(df)
    return df

# (countrycode) 10digitnumber, 2362


def standardize_phone(phone):
    ext_split = re.split(r'\s*(?:extn?\.?|,)\s*',
                         str(phone), flags=re.IGNORECASE)
    main_number = ext_split[0]
    extension = ext_split[1] if len(ext_split) > 1 else ""
    digits = re.sub(r'\D', '', main_number)
    extension_digits = re.sub(r'\D', '', extension)
    standardized_phone = '+' + digits
    if extension_digits:
        standardized_phone += f" Ext {extension_digits}"
    return standardized_phone


def standardize_email(email):
    email = re.sub(r'\s+at\s+', '@', str(email), flags=re.IGNORECASE)
    email = re.sub(r'(?<=@[\w.]+)', '.com',
                   email) if '@' in email and '.' not in email.split('@')[1] else email
    return email

# mm-dd-yy= yy-mm-dd Jan 01/


def standardize_date(date):
    return parser.parse(str(date)).strftime('%Y-%m-%d')


def Standardize(df, standards):
    for column, format_type in standards.items():
        if column in df.columns:
            if format_type == "phone number format" and df[column].astype(str).str.contains(r'\+?\d{1,4}?[-. ]?\(?\d{1,3}?\)?[-. ]?\d{1,4}[-. ]?\d{1,9}').any():
                df[column] = df[column].apply(standardize_phone)
            elif format_type == "email format" and df[column].astype(str).str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b').any():
                df[column] = df[column].apply(standardize_email)
            elif format_type == "date format" and pd.to_datetime(df[column], errors='coerce').notna().any():
                df[column] = df[column].apply(standardize_date)
    return df


def StringMatching(df, variations):
    for column, targets in variations.items():
        if column in df.columns:
            for target, variations in targets.items():
                df[column] = df[column].replace(variations, target)
    return df
