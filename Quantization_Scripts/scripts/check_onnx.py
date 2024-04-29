import onnx

onnx_file_path = "/workspace/QualcommClinic23-1/Quantization_Scripts/onnx_files/roberta-base-squad2.onnx"

onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)

input_names = [input.name for input in onnx_model.graph.input]
output_names = [output.name for output in onnx_model.graph.output]
print(input_names)
print(output_names)