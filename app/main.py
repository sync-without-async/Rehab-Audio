from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import *

from fastapi.testclient import TestClient

from connector import database_connector, database_query

import logging

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = TestClient(app)

logging.basicConfig(level=logging.INFO, format='[DSR_MODULE]%(asctime)s %(levelname)s %(message)s')

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):   return PlainTextResponse(str(exc), status_code=400)

@app.exception_handler(Exception)
async def http_exception_handler(request, exc): return PlainTextResponse(str(exc), status_code=500)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/getSummary")
def getSummary():
    # TODO: get audio from database
    connector, cursor = database_connector(database_secret_path="secret_key.json")
    table_name = "video"
    query = f"SELECT * FROM {table_name}"
    result = database_query(connector, cursor, query, verbose=False)

    return result

    # TODO: denoising

    # TODO: ASR, STT

    # TODO: Get summary

    # TODO: Insert summary to database

    # TODO: Return end flag

def test_read_main():
    response = client.get("/getSummary")
    assert response.status_code == 200
    print(response.json())

if __name__ == "__main__":
    test_read_main()

