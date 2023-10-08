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
import asyncio
import torch

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
def getSummary(ano: int):
    connector, cursor = database_connector(database_secret_path="secret_key.json")
    table_name = "audio"
    query = f"SELECT * FROM {table_name}"
    result = database_select_using_pk(
        table=pl.DataFrame(database_query(connector, cursor, query, verbose=False)),
        pk=ano,
        verbose=True
    )

    result = result.to_numpy().tolist()[0]

    try:
        doctor_audio_url, patient_audio_url = result[2], result[5]
        doctor_audio = requests.get(doctor_audio_url).content
        patient_audio = requests.get(patient_audio_url).content

        with open("doctor.wav", "wb") as f:     f.write(doctor_audio)
        with open("patient.wav", "wb") as f:    f.write(patient_audio)
        doctor_audio, doc_fs = den.load_audio("doctor.wav")
        patient_audio, pat_fs = den.load_audio("patient.wav")

        asyncio.run(_do_summary(
            ano=ano,
            doctor_audio=doctor_audio,
            patient_audio=patient_audio,
            doc_fs=doc_fs,
            pat_fs=pat_fs,
        ))

        return True

    except Exception as e:
        logging.error(e)
        return False

async def _do_summary(
        ano: int, 
        doctor_audio: torch.Tensor,
        patient_audio: torch.Tensor,
        doc_fs: int,
        pat_fs: int,
    ):
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
        max_tokens=1024,
        verbose=True,
    )

    connector, cursor = database_connector(database_secret_path="secret_key.json")
    table_name, table_column = "audio", "summary"

    _ = insert_summary_database(
        connector=connector,
        cursor=cursor,
        target_table_name=table_name,
        target_columns=table_column,
        target_values=summarized,
        target_room_number=ano,
        verbose=True,
    )
