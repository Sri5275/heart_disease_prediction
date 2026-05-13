# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
import pickle
import eli5
from eli5.sklearn import PermutationImportance


# Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# Extract the Cleveland dataset for analysis, as it is the most commonly used dataset for heart disease prediction and has the most complete data
cldf = df[df["dataset"]=="Cleveland"]

cldf = cldf.drop(columns=["dataset"]) # we will only use the Cleveland dataset for our analysis, so we can drop the "dataset" column
cldf["num"] = (cldf["num"]>0).astype(int) # originally num is 0-4, we will convert it to 0 and 1, where 1 means heart disease is present and 0 means heart disease is not present
cldf = cldf.dropna() # drop any rows with missing values


# %%
print(cldf.head())
# print(cldf["cp"].value_counts())
# Converting categorical variables into dummy/indicator variables using one-hot encoding, and dropping the first category to avoid multicollinearity in the logistic regression model
cldf = pd.get_dummies(cldf, columns=["sex","cp", "restecg", "slope", "thal"], drop_first=True) # convert categorical variables into dummy/indicator variables, and drop the first category to avoid multicollinearity

# print(cldf.info())
# print(cldf.isnull().sum()) 
# print(cldf.describe())
# print(cldf["num"].value_counts())

# Visualize the distribution of the target variable "num" using a count plot, which shows the count of each class (0 and 1) in the target variable
sns.countplot(x="num",data=cldf) # visualize the distribution of the target variable "num"
plt.show()  


# for col in cldf.columns:
#     print(cldf[col].value_counts()) # print the value counts for each column to understand the distribution of the data and identify any potential issues such as class imbalance or missing values

# Drop the "num" column from the features and use it as the target variable for our machine learning model. The "num" column indicates the presence of heart disease, where 1 means heart disease is present and 0 means heart disease is not present.
X = cldf.drop(columns=["num"]) # features
y = cldf["num"] # target variable

# Split the data into training and testing sets, with 20% of the data used for testing and a random state of 42 for reproducibility. We also use stratify=y to ensure that the class distribution in the training and testing sets is similar to the original dataset, which is important for imbalanced datasets like this one.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    ) # split the data into training and testing sets, with 20% of the data used for testing and a random state of 42 for reproducibility

# break 1: X_test = X_test.drop("age",axis=1) # drop the "age" column from the test set, as it is not needed for prediction and may cause issues with the model

# Scaling the features using StandardScaler, which standardizes the features by removing the mean and scaling to unit variance. This is important for many machine learning algorithms, including logistic regression, as it can improve the performance of the model and ensure that all features are on the same scale.
scaler = StandardScaler().set_output(transform="pandas") # create a standard scaler object and set the output to be a pandas DataFrame, which allows us to easily work with the scaled features and maintain the column names for better interpretability.
X_train = scaler.fit_transform(X_train) # fit the scaler to the training features and transform them to have a mean of 0 and a standard deviation of 1
X_test = scaler.transform(X_test) # transform the test features using the same scaler (note: we should use the same scaler for both training and testing data to ensure consistency)
# print(X_test[:5])

# Create a logistic regression model and fit it to the training data. We set max_iter=2000 to ensure that the model converges, as logistic regression can sometimes require more iterations to converge, especially with larger datasets or when the features are not well-scaled.
model = LogisticRegression(max_iter = 2000) # create a logistic regression model
model.fit(X_train,y_train) # fit the model to the training data

# Predict the probabilities of the positive class (heart disease present) for the test set using the predict_proba method of the logistic regression model. 
# %%
y_pred = model.predict(X_test) # make predictions on the test set
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Classification Report:\n", classification_report(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))

# Save the trained model to a file using pickle, which allows us to save the model and its parameters for later use without having to retrain it. 
# We save both the scaler and the model together in a tuple, so that we can easily load them both when we want to make predictions on new data.
with open("heart_disease_model.pkl","wb") as f:
    pickle.dump((scaler,model),f) # save the trained model to a file using pickle

# Sample prediction: we take the first 4 samples from the test set and make predictions using the trained model. 
# We also print the actual labels for these samples to compare with the predictions and evaluate the performance of the model on these specific samples.
sample = X_test[0:4] # take the first sample from the test set
print("Prediction for the sample:", model.predict(sample)) # make a prediction for the sample using the trained model   
print("Actual label for the sample:", y_test.iloc[0:4].values) # print the actual label for the sample to compare with the prediction

# Calculate the predicted probabilities for the positive class (heart disease present) for the test set using the predict_proba method of the logistic regression model.
# ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) is a performance metric for binary classification models that measures the ability of the model to distinguish between the positive and negative classes.
y_prob = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Calculate the permutation importance of the features using the PermutationImportance class from the eli5 library. 
# This method evaluates the importance of each feature by measuring the decrease in model performance (e.g., accuracy) when the values of that feature are randomly permuted.
perm=PermutationImportance(model, random_state=1).fit(X_test,y_test)

# Display the feature importance weights using the show_weights function from the eli5 library. 
# This will show the importance of each feature in the model, with higher weights indicating more important features for predicting heart disease.
# %%
eli5.show_weights(perm,feature_names=X_test.columns.tolist())

# Display the predictions for the first 4 samples in the test set using the show_predictions function from the eli5 library. 
# This will show the predicted probabilities for each class (heart disease present and not present) for these samples
# %%
eli5.show_prediction(model, X_test.iloc[0], feature_names=X_test.columns.tolist())

# %%
