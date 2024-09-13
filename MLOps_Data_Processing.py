import pandas as pd
from dataprep.datasets import load_dataset
from dataprep.eda import create_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, classification_report
import lime
import lime.lime_tabular
import pickle


def load_and_preprocess_data():
    df = load_dataset("titanic")
    #Perform AutoEDA using dataprep and save the report as HTML
    report = create_report(df)
    report.save("auto_eda_report.html")

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Drop unnecessary columns
    df.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    # Feature engineering
    df['Title'] = df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df.drop(['Name'], axis=1, inplace=True)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

    # Ensure all expected features are present
    expected_columns = ['Pclass', 'Age', 'Fare', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S',
                         'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other', 'Title_the Countess']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value

    # Scaling numerical features
    scaler = StandardScaler()
    df[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(df[['Age', 'Fare', 'FamilySize']])

    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return df


def split_data(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def model_selection(X_train, y_train):
    tpot = TPOTClassifier(
        generations=5,
        population_size=20,
        random_state=42,
        config_dict='TPOT sparse',
        verbosity=2,
        n_jobs=-1
    )
    tpot.fit(X_train, y_train)
    with open('best_pipeline.pkl', 'wb') as f:
        pickle.dump(tpot.fitted_pipeline_, f)
    print("Best pipeline found by TPOT:")
    print(tpot.fitted_pipeline_)
    return tpot


def evaluate_model(tpot, X_test, y_test):
    y_pred = tpot.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy of the best TPOT model: {accuracy}")
    print(classification_report(y_test, y_pred))


def export_pipeline(tpot):
    tpot.export('best_pipeline.py')


def implement_xai(X_train, X_test, tpot):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=['Not Survived', 'Survived'],
        mode='classification'
    )

    i = 0  # index of the instance to explain
    exp = explainer.explain_instance(X_test.iloc[i].values, tpot.fitted_pipeline_.predict_proba)
    exp.save_to_file('lime_explanation.html')


def main():
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(df)
    tpot = model_selection(X_train, y_train)
    evaluate_model(tpot, X_test, y_test)
    export_pipeline(tpot)
    implement_xai(X_train, X_test, tpot)

    print(X_train.columns.tolist())


if __name__ == "__main__":
    main()
