# train_model.py
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("C:/Users/JOMON/Desktop/spam email/project/spam.csv", encoding='latin-1')
data=data[['v1','v2']]
data.columns=['label','message']

data['label']=data['label'].map({'ham':0,'spam':1})

x_train,x_test,y_train,y_test=train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    stratify=data['label'],
    random_state=42

)


vectorizer=TfidfVectorizer(
    stop_words='englis',
    ngram_range=(1,2)
)

X_train_tfidf=vectorizer.fit_transform(x_train)
x_test_tfidf=vectorizer.transform(x_test)


model=LogisticRegression(class_weight='balanced',max_iter=1000)
model.fit(X_train_tfidf, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Model Saved Successfully!")