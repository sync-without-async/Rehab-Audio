from huggingface_hub import HfApi

from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer
)

import torch

# HuggingFace Login

# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME = "daryl149/llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)

model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    return_dict=True,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

prompt = "Hello, how are you?"
encoded_prompt = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        max_new_tokens=256,
        temperature=0,
        generation_config=generation_config,
        **encoded_prompt,
    )

decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
print(decoded_output)
