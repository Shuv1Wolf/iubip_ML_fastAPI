import pickle

import nltk
import pandas as pd
import pymorphy2
from fastapi import FastAPI, HTTPException
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from starlette.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI()

origins = [
    "http://localhost",
    "http://90.156.210.55",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Access-Control-Allow-Origin",
                   "Access-Control-Allow-Methods", "X-Requested-With",
                   "Authorization", "X-CSRF-Token"]
)

nltk.download("stopwords")
nltk.download('punkt')

df = pd.read_csv('train_data_preprocessed.csv', sep=',')
X_train, X_test = train_test_split(df['utterance'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    morph = pymorphy2.MorphAnalyzer()
    text = [morph.parse(word)[0].normal_form for word in filtered_tokens]

    return " ".join(text)


def ml(text, file_name):
    model = pickle.load(open(file_name, 'rb'))

    user_input = preprocess_text(text)
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    return prediction[0]


@app.post("/api/ml_model")
async def root(utterance: str):
    try:
        passive_request = ml(utterance, 'ml/svm_request.pickle')
        passive_importance = ml(utterance, 'ml/сlassifier_importance.pickle')
        return {
            'category': passive_request,
            'importance': passive_importance
        }
    except Exception:
        raise HTTPException(status_code=500, detail={
            "status": "error",
            "data": None,
            "details": 'ML error or preprocessor error'
        })


Instrumentator().instrument(app).expose(app)

