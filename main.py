from fastapi import FastAPI, Form, HTTPException, UploadFile, Query
from mpstemmer import MPStemmer
from numpy import argsort
from polars import read_csv, Series
from PyPDF2 import PdfReader
from re import sub
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = read_csv("temp/repository_pnj_20212023clean.csv")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["f"].to_list())


def find_similar_documents(title: str, abstract: str, top_n: int):
    cosine_scores = cosine_similarity(
        vectorizer.transform(
            [
                StopWordRemoverFactory()
                .create_stop_word_remover()
                .remove(
                    MPStemmer().stem_kalimat(
                        " ".join(
                            sub(
                                r"[^a-z]", " ", (title + " " + abstract).lower()
                            ).split()
                        )
                    )
                )
            ]
        ),
        tfidf_matrix,
    ).flatten()

    top_n_indices = argsort(cosine_scores)[-top_n:][::-1]

    return (
        df[top_n_indices]
        .with_columns(
            Series(name="similarity", values=cosine_scores[top_n_indices] * 100)
        )
        .drop("f")
    )


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
        return find_similar_documents(
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
        return find_similar_documents(
            pages[0].extract_text(),
            " ".join([page.extract_text() for page in pages[1:]]),
            top_n,
        ).to_dicts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
