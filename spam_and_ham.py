#Python program used to classify text sms into ham or spam with machine learning.(NLTK)
#Author - Hitesh Thakur

#Imported reuired libraries
import pickle;
import pandas as pd;
from sklearn.svm import SVC;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;
from sklearn.feature_extraction.text import TfidfVectorizer;
from sklearn.metrics import confusion_matrix,accuracy_score;
from nltk.tokenize import word_tokenize;
from  nltk.corpus import stopwords;
from nltk.stem import PorterStemmer;

#Reading .csv file containing training data using pandas
dataset = pd.read_csv("spam.csv",encoding='latin-1');
raw_X = dataset.iloc[:,1];      #Feature
raw_Y = dataset.iloc[:,0];      #Label

#Using NLTK libraries function to remove stioop words and to perform stemming
def word_token(text):
    text=text.lower();
    wordset=word_tokenize(text);
    stop_words=set(stopwords.words('english'));
    filtered_words=[];
    stemming = PorterStemmer();
    for x in wordset:
        if x not in stop_words:
            filtered_words.append(stemming.stem(x));
    text=" ".join(filtered_words)
    return text;

#Calling word_token function for each sample in the dataset
X=raw_X.apply(word_token);

#Conerting non numeric data into numeric data using skleran Label Encoder
le=LabelEncoder();
Y=le.fit_transform(raw_Y);

#Splitting dataset into two parts: Train and Test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=0);

#With the help of TfidfVectorizer- Transformed text to feature vectors that can be used as input to estimator 
tfidf=TfidfVectorizer();
X_train=tfidf.fit_transform(X_train);
X_test=tfidf.transform(X_test);

#Using SVM as a model to classify and trained it using out data generated from TFIDF 
svc=SVC(kernel='linear');
svc.fit(X_train,Y_train);

#Predicting labels for the test dataset
Y_pred=svc.predict(X_test);
con_mat=confusion_matrix(Y_pred,Y_test);
accuracy=accuracy_score(Y_pred,Y_test);

#Printing the accuracy and the confusion matrix for the trained SVM model 
print("Confusion Matrix :\n",con_mat);
print("Accuracy : ",accuracy);
