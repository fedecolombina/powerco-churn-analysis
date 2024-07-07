import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data_part1_path, data_part2_path, output_path, is_train=True, output_template_path=None):

    data_part1 = pd.read_csv(data_part1_path)
    data_part2 = pd.read_csv(data_part2_path)

    if is_train:
        data_part3 = pd.read_csv('data/raw/ml_case_training_output.csv')
    else:
        data_part3 = pd.read_csv(output_template_path)

    #drop the index column
    if 'Unnamed: 0' in data_part1.columns:
        data_part1.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in data_part2.columns:
        data_part2.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in data_part3.columns:
        data_part3.drop(columns=['Unnamed: 0'], inplace=True)

    #take mean in data_part2 to have one row per 'id'
    data_part2_agg = data_part2.groupby('id').agg({
        'price_p1_var': ['mean', 'std', 'min', 'max'],
        'price_p2_var': ['mean', 'std', 'min', 'max'],
        'price_p3_var': ['mean', 'std', 'min', 'max'],
        'price_p1_fix': ['mean', 'std', 'min', 'max'],
        'price_p2_fix': ['mean', 'std', 'min', 'max'],
        'price_p3_fix': ['mean', 'std', 'min', 'max']
    }).reset_index()

    #flatten the column index from aggregation
    data_part2_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data_part2_agg.columns.values]

    #merge csv inputs into one file
    data = data_part1.merge(data_part2_agg, left_on='id', right_on='id_', how='left').merge(data_part3, on='id', how='left')
    data.drop(columns=['id_'], inplace=True)  #drop the redundant 'id_' column

    #remove duplicates if any
    data = data.drop_duplicates()

    #don't apply preprocessing to id and churn (already 0 or 1)
    id_column = data['id']
    if is_train:
        churn_column = data['churn']
        data.drop(columns=['id', 'churn'], inplace=True)
    else:
        data.drop(columns=['id'], inplace=True)

    #fix missing values
    data = data.dropna(thresh=len(data) * 0.5, axis=1) #drop columns with >50% empty rows (otherwise values don't make sense)

    #separate numeric and non-numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns

    #numeric columns: fill missing values with median
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    #non-numeric columns: fill missing values with mode
    for col in non_numeric_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    #encode non-numeric variables
    label_encoders = {}
    for col in non_numeric_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    #scale numerical features
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    #add 'id' and 'churn' columns back to the data
    data['id'] = id_column
    if is_train:
        data['churn'] = churn_column

    #put 'id' as the first column
    columns = ['id'] + [col for col in data if col != 'id']
    data = data[columns]

    data.to_csv(output_path, index=False)

def main():

    #training data
    preprocess_data(
        'data/raw/ml_case_training_data.csv',
        'data/raw/ml_case_training_hist_data.csv',
        'data/processed/train_data.csv',
        is_train=True
    )

    #testing data
    preprocess_data(
        'data/raw/ml_case_test_data.csv',
        'data/raw/ml_case_test_hist_data.csv',
        'data/processed/test_data.csv',
        is_train=False,
        output_template_path='data/raw/ml_case_test_output_template.csv'
    )

if __name__ == "__main__":
    main()
