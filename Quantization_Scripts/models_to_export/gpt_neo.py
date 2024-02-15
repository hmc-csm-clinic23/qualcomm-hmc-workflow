# import torch
# from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# model_name = "EleutherAI/gpt-neo-125M"
# model = GPTNeoForCausalLM.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# # Create a dummy input for the model
# input_text = "Hello, world!"
# encoded_input = tokenizer(input_text, return_tensors="pt")
# input_ids = encoded_input["input_ids"]
# attention_mask = encoded_input["attention_mask"]

# # Specify the path for the output ONNX file
# onnx_file_path = "gpt-neo-125M.onnx"

# # Export the model with attention_mask
# torch.onnx.export(model,                               # model being run
#                   (input_ids, attention_mask),         # model input (or a tuple for multiple inputs)
#                   onnx_file_path,                      # where to save the model
#                   opset_version=11,                    # the ONNX version to export the model to
#                   input_names=['input_ids', 'attention_mask'],   # the model's input names
#                   output_names=['output'],             # the model's output names
#                   dynamic_axes={'input_ids': {0: 'batch_size'},    # variable length axes
#                                 'attention_mask': {0: 'batch_size'},
#                                 'output': {0: 'batch_size'}})

import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model_name = "EleutherAI/gpt-neo-125M"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Create a dummy input for the model
input_text = "Hello, world!"
encoded_input = tokenizer(input_text, return_tensors="pt")
input_ids = encoded_input["input_ids"]

# Specify the path for the output ONNX file
onnx_file_path = "gpt-neo-125M.onnx"

# Export the model without attention_mask
torch.onnx.export(model,                               # model being run
                  input_ids,                           # model input (or a tuple for multiple inputs)
                  onnx_file_path,                      # where to save the model
                  opset_version=11,                    # the ONNX version to export the model to
                  input_names=['input_ids'],           # the model's input names
                  output_names=['output'],             # the model's output names
                  dynamic_axes={'input_ids': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})