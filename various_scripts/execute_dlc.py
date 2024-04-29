from pysnpe_utils.pysnpe import SnpeContext, TargetDevice
from pysnpe_utils.pysnpe_enums import ModelFramework, DeviceType, DlcType
import numpy as np

frozen_graph_path = "./frozen_models/distilbert.pb"

dlc_path = "./dlc_files/distilbert_using_pyscript.dlc"

MAX_SEQ_LENGTH = 128

input_tensor_map = {
    "input_ids" : [1, MAX_SEQ_LENGTH],
    "attention_mask" : [1, MAX_SEQ_LENGTH]
}

output_tensor_names = ["Identity:0", "Identity_1:0"]

target_device = TargetDevice(
    target_device_type= DeviceType.X86_64_LINUX
)

snpe_context = SnpeContext(
    model_path=frozen_graph_path,
    model_framework=ModelFramework.TF,
    dlc_path=dlc_path,
    input_tensor_map=input_tensor_map,
    output_tensor_names=output_tensor_names,
    target_device=target_device
)

# snpe_context.to_dlc()

import tensorflow as tf

from transformers import TensorType
from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer
import sys

bs = 1
SEQ_LEN = 128
MODEL_NAME = "distilbert-base-cased-distilled-squad"

# Allocate tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Sample input
context = "The government of France is based in Paris. Another famous city is Nice, in the south of France."
question = "Describe France?"

input_encodings = tokenizer(
            question,
            context,
            return_tensors=TensorType.TENSORFLOW,
            # return_tensors="np",
            padding='max_length',
            return_length=True,
            max_length=SEQ_LEN,
            return_special_tokens_mask=True
        )

input_tensor_map = {
    "input_ids:0": input_encodings.input_ids.numpy(),
    "attention_mask:0": input_encodings.attention_mask.numpy()
}

output_tensor = snpe_context.execute_dlc(
    input_tensor_map=input_tensor_map,
    dlc_type=DlcType.FLOAT,
    target_device=target_device
)

# print(output_tensor)

# Assuming output is the dictionary returned by execute_dlc
start_logits = output_tensor['Identity_1:0']
end_logits = output_tensor['Identity:0']

# Apply softmax to convert logits to probabilities
start_probs = np.exp(start_logits) / np.sum(np.exp(start_logits), axis=-1, keepdims=True)
end_probs = np.exp(end_logits) / np.sum(np.exp(end_logits), axis=-1, keepdims=True)

# Find the indices of the maximum probabilities
# answer_start_index = np.argmax(start_probs)
# answer_end_index = np.argmax(end_probs)
answer_start_index = int(tf.math.argmax(start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(end_logits, axis=-1)[0])

predict_answer_tokens = input_encodings.input_ids[0, answer_start_index : answer_end_index + 1]
ans = tokenizer.decode(predict_answer_tokens)
print(type(ans[0]))

print("Predicted answer:", ans)
