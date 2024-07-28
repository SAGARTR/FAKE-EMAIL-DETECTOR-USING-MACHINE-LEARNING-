# FAKE-EMAIL-DETECTOR-USING-MACHINE-LEARNING-
Developed a spam detection model using logistic regression, achieving high accuracy by  leveraging TF-IDF vectorization and label encoding techniques.



Phishing detection is critical in cyber security to identify and mitigate fraudulent activities. This practical implementation demonstrates the use of Logistic Regression, a supervised learning algorithm, to classify emails as spam or ham (legitimate).

Key Components of Practical Implementation:

1. Loading the Dataset: The dataset is loaded using `pandas`. We assume the dataset `mail_data.csv` contains email messages categorized as 'spam' or 'ham'.

   import pandas as pd
   df = pd.read_csv('mail_data.csv')
   print(df.head())

2. Handling Missing Values: Replace any missing values in the dataset with empty strings.

   data = df.where((pd.notnull(df)), '')

3. Data Inspection: Inspect the dataset to understand its structure and size.

   print(data.info())
   print(data.shape)
   
4. Label Encoding: Convert the 'Category' column to numerical values where 'spam' is 0 and 'ham' is 1.

   data.loc[data['Category'] == 'spam', 'category'] = 0
   data.loc[data['Category'] == 'ham', 'category'] = 1

5. Extracting Features and Labels: Separate the message content and the labels.

   X = data['Message']
   Y = data['Category']

6. Splitting the Dataset: Split the dataset into training and testing sets (80% training, 20% testing).

   from sklearn.model_selection import train_test_split
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

7. TF-IDF Vectorization: Transform the text data into numerical features using TfidfVectorizer.

   from sklearn.feature_extraction.text import TfidfVectorizer
   feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
   X_train_features = feature_extraction.fit_transform(X_train)
   X_test_features = feature_extraction.transform(X_test)

8. Label Encoding: Encode the target labels as numerical values.

   from sklearn.preprocessing import LabelEncoder
   label_encoder = LabelEncoder()
   Y_train_encoded = label_encoder.fit_transform(Y_train)
   Y_test_encoded = label_encoder.transform(Y_test)

9. Training the Model: Train the Logistic Regression model on the training data.

   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression()
   model.fit(X_train_features, Y_train_encoded)

10. Accuracy on Training Data: Evaluate the model's accuracy on the training data.

    from sklearn.metrics import accuracy_score
    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(Y_train_encoded, prediction_on_training_data)
    print('Accuracy on training data:', accuracy_on_training_data)

11. Accuracy on Testing Data: Evaluate the model's accuracy on the testing data.

    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test_encoded, prediction_on_test_data)
    print('Accuracy on testing data:', accuracy_on_test_data)

12. Predicting New Data: Predict the category of a new email message.

    input_your_mail = ["Your input email message here"]
    input_data_features = feature_extraction.transform(input_your_mail)
    prediction = model.predict(input_data_features)
    print(prediction)
