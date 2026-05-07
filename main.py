# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

df = pd.read_csv('heart_disease_uci.csv')

cldf = df[df["dataset"]=="Cleveland"]

cldf = cldf.drop(columns=["dataset"]) # we will only use the Cleveland dataset for our analysis, so we can drop the "dataset" column
cldf["num"] = (cldf["num"]>0).astype(int) # originally num is 0-4, we will convert it to 0 and 1, where 1 means heart disease is present and 0 means heart disease is not present
cldf = cldf.dropna() # drop any rows with missing values

# print(cldf.head())
# print(cldf["cp"].value_counts())
cldf = pd.get_dummies(cldf, columns=["sex","cp", "restecg", "slope", "thal"], drop_first=True) # convert categorical variables into dummy/indicator variables, and drop the first category to avoid multicollinearity

# print(cldf.info())
# print(cldf.isnull().sum())
# print(cldf.describe())
# print(cldf["num"].value_counts())

# %%
sns.countplot(x="num",data=cldf) # visualize the distribution of the target variable "num"
plt.show()  


# for col in cldf.columns:
#     print(cldf[col].value_counts()) # print the value counts for each column to understand the distribution of the data and identify any potential issues such as class imbalance or missing values


X = cldf.drop(columns=["num"]) # features
y = cldf["num"] # target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    ) # split the data into training and testing sets, with 20% of the data used for testing and a random state of 42 for reproducibility

# break 1: X_test = X_test.drop("age",axis=1) # drop the "age" column from the test set, as it is not needed for prediction and may cause issues with the model

scaler = StandardScaler() # create a standard scaler object
X_train = scaler.fit_transform(X_train) # fit the scaler to the training features and transform them to have a mean of 0 and a standard deviation of 1
X_test = scaler.transform(X_test) # transform the test features using the same scaler (note: we should use the same scaler for both training and testing data to ensure consistency)
# print(X_test[:5])

model = LogisticRegression(max_iter = 1000) # create a logistic regression model
model.fit(X_train,y_train) # fit the model to the training data

y_pred = model.predict(X_test) # make predictions on the test set
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Classification Report:\n", classification_report(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))


with open("heart_disease_model.pkl","wb") as f:
    pickle.dump(model,f) # save the trained model to a file using pickle

sample = X_test[0:4] # take the first sample from the test set
print("Prediction for the sample:", model.predict(sample)) # make a prediction for the sample using the trained model   
print("Actual label for the sample:", y_test.iloc[0:4].values) # print the actual label for the sample to compare with the prediction


#Step 10: Break the system, build a better model, and repeat the process

