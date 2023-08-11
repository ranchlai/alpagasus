# """Inference with llama-alpaca"""
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from accelerate import init_empty_weighs, infere

# # model_name = "../models/alpagasus-7b"
# model_name = "../models/open_llama_7b_v2/"


# tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map

# prompt = "The following is a set of instructions for a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"

# inputs = tokenizer(prompt, return_tensors="pt")
# outputs = model.generate(inputs["input_ids"].to("cuda"), max_length=512, num_beams=5, num_return_sequences=5, temperature=0.9)

# responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
# print(responses)


import torch
from transformers import LlamaTokenizer, LlamaForCausalLM ,AutoConfig
from accelerate import init_empty_weights,infer_auto_device_map
from accelerate.utils import get_balanced_memory
## v2 models
model_path = '../models/open_llama_7b_v2'

## v1 models
# model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'


tokenizer = LlamaTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
with init_empty_weights():
    model = LlamaForCausalLM(config)
    
max_memory = get_balanced_memory(
    model,
    max_memory=None,
    dtype='float16',
    low_zero=False,
)

device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    dtype='float16',
    verbose=True,
    no_split_module_classes=["LlamaDecoderLayer"],
    
)
print(device_map)

# import code; code.interact(local=dict(globals(), **locals()))
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map=device_map,
)

# print dtype and device
for name, param in model.named_parameters():
    print(name, param.dtype, param.device)
    
# import code; code.interact(local=dict(globals(), **locals()))

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))

