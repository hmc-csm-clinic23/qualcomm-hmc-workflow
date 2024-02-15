import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()

# Prepare a dummy input with static dimensions
question = "What is the capital of France?"
context = "Paris is the capital of France."
inputs = tokenizer(question, context, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
print(inputs)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Specify the path for the output ONNX file
onnx_file_path = "roberta_base_squad2_ids.onnx"

# Export the model with static input dimensions
torch.onnx.export(model,                                        # model being run
                  (input_ids, attention_mask),                  # model input (or a tuple for multiple inputs)
                  onnx_file_path,                               # where to save the model
                  opset_version=11,                             # the ONNX version to export the model to
                  input_names=['input_ids', 'attention_mask'],  # the model's input names
                  output_names=['Identity:0', 'Identity_1:0'],  # the model's output names
                  dynamic_axes=None)                            # Remove dynamic axes