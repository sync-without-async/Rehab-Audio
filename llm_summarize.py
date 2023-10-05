from transformers import (
    AutoTokenizer,
    GenerationConfig,

    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM, 
)

import transformers
import torch

import logging

logging.basicConfig(level=logging.INFO, format="[SUMMARY_MODULE]%(asctime)s %(levelname)s %(message)s")

MODEL_NAME = "daryl149/llama-2-7b-chat-hf"
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    return_dict=True,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

def _get_system_prompt(
        system_prompt: str = None,
        user_prompt: str = None,
        verbose: bool = False,
    ) -> str:
    if verbose:
        logging.info(f"system_prompt: {system_prompt}")
        logging.info(f"user_prompt: {user_prompt}")
    
    if type(system_prompt) == list:
        retyped_system_prompt = str() 
        for text in system_prompt:
            retyped_system_prompt += text
        system_prompt = retyped_system_prompt

    if type(user_prompt) == list:
        retyped_user_prompt = str() 
        for text in user_prompt:
            retyped_user_prompt += text
        user_prompt = retyped_user_prompt

    prompt = system_prompt + "\n" + user_prompt
    prompt = prompt.strip()
    if verbose: logging.info(f"prompt: {prompt}")

    return prompt

def _combine_doctor_patient_prompt(
        doctor_prompt: list = None,
        patient_prompt: list = None,
        verbose: bool = False,
    ) -> str:
    doctor_prompt = doctor_prompt[0].strip()
    patient_prompt = patient_prompt[0].strip()
    
    prompt = f"""DOC:<{doctor_prompt}>PATIENT:<{patient_prompt}>"""
    if verbose: logging.info(f"prompt: {prompt}")

    return prompt

def generate_response(
        prompt: str, 
        max_new_tokens: int = 128
    ) -> str:
    encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            temperature=0,
            generation_config=generation_config,
        )
    answer_tokens = outputs[:, encoding.input_ids.shape[1] :]
    return tokenizer.decode(answer_tokens[0], skip_special_tokens=True)

def summary(
        doctor_text: str = None,
        patient_text: str = None,
        system_prompt: str = None,
        device: torch.device = None,
        max_length: int = 1024,
        verbose: bool = False,

        prompt: str = None,
    ) -> str:
    if doctor_text is None or patient_text is None: raise ValueError(f"doctor_prompt and patient_prompt must be provided, but doctor prompt is {type(doctor_text)} and patient prompt is {type(patient_text)}")
    if system_prompt is None:
        system_prompt = f"""
당신은 문서 정리를 하는 서기 입니다. 그 중에서도 두 사람의 대화 내용을 듣고 어떤 대화인지 정리 요약하는 서기 입니다. 두 사람의 대화 내용을 당신에게 전달할 것입니다. 한명은 의사, 한명은 환자입니다. 의사는 DOC:<TEXT HERE>로 드릴 것이며 환자는 PATIENT:<TEXT HERE>로 드릴 예정입니다. 대화 내용에서는 시간적 특성이 배제 되어 있습니다. 당신은 별 다른 시간 인덱스가 없더라도 내용을 파악하고 이해하셔야 합니다. 요약 정리는 Markdown 문서 형식으로 정리가 되어야합니다. 서론, 본론, 결론으로 정리를 해야합니다. 서론은 #서론, 본론은 #본론, 결론은 #결론으로 정리하시기 바랍니다.
"""
    if device is None: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device is not provided. Using {device} as default.")

    if verbose: logging.info(f"{MODEL_NAME} is loaded on {device}") 
    if verbose: logging.info(f"Loading prompt...")

    if prompt is None:
        doctor_prompt = doctor_text.split("\n")
        patient_prompt = patient_text.split("\n")
        system_prompt = system_prompt.split("\n")

        prompt = _get_system_prompt(
            system_prompt=system_prompt, 
            user_prompt=_combine_doctor_patient_prompt(
                doctor_prompt=doctor_prompt, 
                patient_prompt=patient_prompt),
            verbose=verbose)
        if verbose: logging.info(f"prompt type: {type(prompt)}")

        if verbose: logging.info(f"Encoding prompt...")
        encoded_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)

    elif prompt is not None:
        if verbose: logging.info(f"prompt type: {type(prompt)}")
        if verbose: logging.info(f"Encoding prompt...")
        encoded_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if verbose: logging.info(f"Generating response...")
    generated_response = generate_response(
        prompt=encoded_prompt,
        max_new_tokens=max_length,
    )

    if verbose: logging.info(f"Generated response: {generated_response}")
    return generated_response 