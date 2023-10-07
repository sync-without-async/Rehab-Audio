# Rehab Machine Learning - Conversation Summary

This repository focuses on the role of artificial intelligence in Project "**Rehab**" for remote healthcare. After concluding remote healthcare sessions, the artificial intelligence system processes the audio from counselors and doctors, performs preprocessing, ASR (Automatic Speech Recognition), and utilizes Prompt Engineered GPT3.5 to provide a summarized conversation.

For details on the remote healthcare functionality, please refer to the content in Rehab Frontend and Rehab Backend.

## Requirements

This code utilizes models from HuggingFace Hub and OpenAI's API. Therefore, you need to have a HuggingFace account, a token, and access to OpenAI's paid API. You will need to log in to your HuggingFace account, and additionally, this code accesses a database to insert summarized results. So, the database information and OpenAI API key should be included in a `secret_key.json` file as shown below:

```json
{
    "OpenAI": {
        "API_KEY": "OPENAI_API_KEY"
    },

    "database": {
        "host": "YOUR_DATABASE_HOST",
        "port": 3306, // Default Port is 3306 in MySQL
        "user": "YOUR_DATABASE_USER",
        "password": "YOUR_DATABASE_USER_PASSWORD",
        "database": "YOUR_DATABASE"
    }
}
```

We use the `kresnik/wav2vec2-large-xlsr-korean` model available on HuggingFace Hub for ASR (Automatic Speech Recognition). Due to ASR performance issues, we are also considering OpenAI's Whisper. To generate summarized features, we use the GPT3.5-Turbo model. System Prompt, User Prompt, and Assistant Prompt are defined and sent to the OpenAI API to organize the content of remote healthcare sessions.

If you want to see the code related to the above description, please refer to `module_demo.ipynb`.

- denoiser (0.1.5 >=)
- numpy (1.23.5 >=)
- torchaudio (2.0.0 >=)
- torch (2.0.0 >=)
- transformers (4.29.2 >=)
- openai (0.28.1 >=)

## Quick Start

To set up the API server for Rehab-ML, we use FastAPI. Therefore, you should have basic FastAPI configurations in place. In actual development, we use [Uvicorn](https://www.uvicorn.org/) as the FastAPI engine. If your terminal's `$PWD` is in the same location as `main.py`, you can start the server with the following command:

```bash
$ uvicorn main:app --host 0.0.0.0 --port 8080
INFO:     Started server process [60991]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

With the above command, the FastAPI server will be accessible via the Public IP Address or localhost on port 8080.