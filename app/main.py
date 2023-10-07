from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import *

from fastapi.testclient import TestClient

from connector import *

import speech_to_text as stt
import denoising as den
import summary

import polars as pl
import requests
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

@app.post("/getSummary")
def getSummary(room_number: int = Form(...)):
    connector, cursor = database_connector(database_secret_path="secret_key.json")
    table_name = "audio"
    query = f"SELECT * FROM {table_name}"
    result = database_select_using_pk(
        table=pl.DataFrame(database_query(connector, cursor, query, verbose=False)),
        pk=room_number,
        verbose=True
    )
    result = result.to_numpy().tolist()[0]

    doctor_audio_url, patient_audio_url = result[2], result[5]
    doctor_audio = requests.get(doctor_audio_url).content
    patient_audio = requests.get(patient_audio_url).content

    with open("doctor.wav", "wb") as f:     f.write(doctor_audio)
    with open("patient.wav", "wb") as f:    f.write(patient_audio)
    
    doctor_audio, doc_fs = den.load_audio("doctor.wav")
    patient_audio, pat_fs = den.load_audio("patient.wav")

    logging.info("[DSR_MODULE] Denoising audio...")
    doctor_audio, doc_sr = den.denoising(
        audio=doctor_audio,
        sample_rate=doc_fs,
        device="cpu",
        verbose=True
    )

    patient_audio, pat_sr = den.denoising(
        audio=patient_audio,
        sample_rate=pat_fs,
        device="cpu",
        verbose=True
    )

    logging.info("[DSR_MODULE] Transcribing audio...") 
    doctor_transcript = stt.speech_to_text(
        processor_pretrained_argument="kresnik/wav2vec2-large-xlsr-korean",
        audio=doctor_audio,
        audio_sample_rate=doc_sr,
        device="cpu",
        verbose=True
    )

    patient_transcript = stt.speech_to_text(
        processor_pretrained_argument="kresnik/wav2vec2-large-xlsr-korean",
        audio=patient_audio,
        audio_sample_rate=pat_sr,
        device="cpu",
        verbose=True
    )

    # TODO: Get summary
    summarized = summary.summarize(
        doctor_content=doctor_transcript,
        patient_content=patient_transcript,
        verbose=True,
    )

    connector, cursor = database_connector(database_secret_path="secret_key.json")
    table_name, table_column = "audio", "summary"

    db_summary_flag = insert_summary_database(
        connector=connector,
        cursor=cursor,
        target_table_name=table_name,
        target_columns=table_column,
        target_values=summarized,
        target_room_number=room_number,
        verbose=True
    )

    return db_summary_flag