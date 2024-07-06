from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from marshal import loads
from mpstemmer import MPStemmer
from numpy import argsort
from polars import read_csv, Series
from pydantic import BaseModel, conint
from PyPDF2 import PdfReader
from re import sub
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from types import FunctionType
from typing import List

# Read the dataset from a CSV file
df = read_csv("temp/repository_pnj_20212023clean.csv")

# Initialize a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the TF-IDF model to the 'f' column of the dataset
tfidf_matrix = vectorizer.fit_transform(df["f"].to_list())

# Get model from a marshal file
with open("temp/model.marshall", "rb") as file:
    model = file.read()

app = FastAPI()


class TextRequest(BaseModel):
    title: str
    abstract: str
    top_n: int


class PDFRequest(BaseModel):
    file: UploadFile
    top_n: int


@app.get("/")
async def get_root():
    return {"message": "Scientia-API by Kelompok 4 TI 6A"}


@app.post("/find_similar")
def post_find_similar(request: TextRequest):
    try:
        if request.top_n <= 0 or request.top_n > df.shape[0]:
            raise HTTPException(
                status_code=400, detail=f"top_n must be between 1 and {df.shape[0]}"
            )

        return FunctionType(loads(model), globals())(
            request.title, request.abstract, request.top_n
        ).to_dicts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/find_similar_pdf")
def post_find_similar_pdf(file: UploadFile, top_n: int = Form(...)):
    try:
        text = PdfReader(file.file).pages[0].extract_text()

        lines = text.split("\n")
        title = lines[0]
        abstract = " ".join(lines[1:])

        if top_n <= 0 or top_n > df.shape[0]:
            raise HTTPException(
                status_code=400, detail=f"top_n must be between 1 and {df.shape[0]}"
            )

        return FunctionType(loads(model), globals())(title, abstract, top_n).to_dicts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
