import pickle;
from nltk.tokenize import word_tokenize;
from  nltk.corpus import stopwords;
from nltk.stem import PorterStemmer;

fh_pk=open('save_model_pickle.sav','rb');
tfidf_from_pickle,svc_from_pickle=pickle.load(fh_pk);

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

message=input("Enter your message");

updated_message=word_token(message);
msg=tfidf_from_pickle.transform([updated_message]);
prediction=svc_from_pickle.predict(msg);

if prediction[0]==0:
    print("Ham");
else:
    print("Spam");