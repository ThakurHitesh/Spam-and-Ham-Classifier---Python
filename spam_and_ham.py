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

dataset = pd.read_csv("spam.csv",encoding='latin-1');
raw_X = dataset.iloc[:,1];
raw_Y = dataset.iloc[:,0];

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

X=raw_X.apply(word_token);

le=LabelEncoder();
Y=le.fit_transform(raw_Y);

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=0);

tfidf=TfidfVectorizer();
X_train=tfidf.fit_transform(X_train);
X_test=tfidf.transform(X_test);

svc=SVC(kernel='linear');
svc.fit(X_train,Y_train);

FH_pk=open("save_model_pickle.sav",'wb');
pickle.dump([tfidf,svc],FH_pk);

Y_pred=svc.predict(X_test);
con_mat=confusion_matrix(Y_pred,Y_test);
accuracy=accuracy_score(Y_pred,Y_test);

print("Confusion Matrix :\n",con_mat);
print("Accuracy : ",accuracy);
