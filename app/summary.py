import openai
import json

import logging

logging.basicConfig(level=logging.INFO, format='[SUMMARY_MODULE]%(asctime)s %(levelname)s %(message)s')

with open("secret_key.json") as f:  secret_key = json.load(f)

openai.api_key = secret_key['OpenAI']['API_KEY']
MODEL_NAME = "gpt-3.5-turbo"

def _get_prompt(
        doctor_content: str = None, 
        patient_content: str = None) -> list:
    if doctor_content is None: raise ValueError(f"doctor_content argument is required. Excepted: str, but got {doctor_content}")

    system_prompt = f"당신은 문서 정리를 하는 서기 입니다. 그 중에서도 두 사람의 대화 내용을 듣고 어떤 대화인지 정리 요약하는 서기 입니다. 두 사람의 대화 내용을 당신에게 전달할 것입니다. 한명은 의사, 한명은 환자입니다. 의사는 DOC:<TEXT HERE>로 드릴 것이며 환자는 PATIENT:<TEXT HERE>로 드릴 예정입니다. 대화 내용에서는 시간적 특성이 배제 되어 있습니다."
    assistant_prompt = "당신은 별 다른 시간 인덱스가 없더라도 내용을 파악하고 이해하셔야 합니다. 요약 정리는 Markdown 문서 형식으로 정리가 되어야합니다. 서론, 본론, 결론으로 정리를 해야합니다. 서론은 #서론, 본론은 #본론, 결론은 #결론으로 정리하시기 바랍니다."
    user_prompt = f"DOC:<{doctor_content}> PATIENT:<{patient_content}>"

    return [
        {"role" : "system", "content" : system_prompt},
        {"role" : "assistant", "content" : assistant_prompt},
        {"role" : "user", "content" : user_prompt},
    ]

def summarize(
        doctor_content: str = None,
        patient_content: str = None,
        verbose: bool = False,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
    """Summarize the conversation between doctor and patient using GPT-3.5 Turbo model.
    System Prompt, Assistant Prompt, and User Prompt will be generated automatically (not automatically, but using get_prompt function).
    This function will return the summary as a string.

    Args:
        doctor_content (str): The content of the doctor.
        patient_content (str): The content of the patient.
        verbose (bool): If True, print the log.
        temperature (float, optional): The temperature of the model. If None, use default value (0.2).
        max_tokens (int, optional): The maximum number of tokens. If None, use default value (250).

    Returns:
        str: The summary as a string."""
    if doctor_content is None: raise ValueError(f"doctor_content argument is required. Excepted: str, but got {doctor_content}")
    if patient_content is None: raise ValueError(f"patient_content argument is required. Excepted: str, but got {patient_content}")
    if temperature is None:
        temperature = 0.2
        logging.info(f"temperature argument is not provided. Using default value: {temperature}")

    if max_tokens is None:
        max_tokens = 250
        logging.info(f"max_tokens argument is not provided. Using default value: {max_tokens}")

    messages = _get_prompt(
        doctor_content=doctor_content,
        patient_content=patient_content,
    )

    if verbose: logging.info(f"Summarizing...")
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response['choices'][0]['message']['content']
