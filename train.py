import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

## loading dataset
print("Loading dataset.....")
df=pd.read_csv("IFND.csv",encoding="latin1")

## cleaning missing values and format target variable 
df=df.dropna(subset=["Statement","Label"])
label_mapping={"TRUE":1,"True":1,"true":1,"Real": 1, 1: 1, "1": 1,"FALSE": 0, "False": 0, "false": 0, "FAKE": 0, "Fake": 0, 0: 0, "0": 0}
df["Label"]=df["Label"].map(label_mapping)

## Dropping any unmapped/corrupted label rows
df=df.dropna(subset=["Label"])
df["Label"]=df["Label"].astype(int)

## separating feature and target
x=df["Statement"].astype(str)
y=df["Label"]

## splitting into train and test sets (80% train, 20% test)
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

## Extract test feature using TF-IDF
print("Vectorizing text....")
vectorizer=TfidfVectorizer(stop_words="english",
                         max_df=0.7,
                         max_features=10000,
                         ngram_range=(1,2))

X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)

## Training Linear SVM model
svm=LinearSVC(C=1.0,dual="auto",random_state=42)
svm.fit(X_train_vec,y_train)

## Evaluating model performance on unseen test data
prediction=svm.predict(X_test_vec)
print("----Classification report(Indian news benchmark)----")
print(classification_report(y_test,prediction,target_names=["Fake","Real"]))

## saving artifacts using pickle
with open("svm_model.pkl","wb") as f:
    pickle.dump(svm,f)

with open("tfidf_vectorizer.pkl","wb") as f:
    pickle.dump(vectorizer,f)

print("saved successfully")