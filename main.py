from fastapi import FastAPI, Form, HTTPException, UploadFile, Query
from marshal import loads
from polars import read_csv
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from types import FunctionType

df = read_csv("temp/repository_pnj_20212023clean.csv")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["f"].to_list())

with open("temp/model.marshall", "rb") as file:
    model = file.read()

app = FastAPI()


@app.get("/")
async def get_root():
    return {"message": "Scientia-API by Kelompok 4 TI 6A"}


@app.get("/find_similar")
async def get_find_similar(
    title: str = Query(...),
    abstract: str = Query(...),
    top_n: int = Query(...),
):
    try:
        return FunctionType(loads(model), globals())(
            title,
            abstract,
            top_n,
        ).to_dicts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/find_similar_pdf")
async def post_find_similar_pdf(file: UploadFile, top_n: int = Form(...)):
    pages = PdfReader(file.file).pages

    try:
        return FunctionType(loads(model), globals())(
            pages[0].extract_text(),
            " ".join([page.extract_text() for page in pages[1:]]),
            top_n,
        ).to_dicts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
