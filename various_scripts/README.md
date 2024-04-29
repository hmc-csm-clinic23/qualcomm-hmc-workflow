## Description
This repository contains the code necessary to export an off-the-shelf machine learning language model as a DLC file, which is then plugged into the 2023-24 Qualcomm CS Clinic team's Android app project. This README file contains the steps necessary to obtain a transformers-based language model (specifically DistilBERT, but the same process can be applied to other models of similar architectures), convert it into a Tensorflow frozen graph, and convert the frozen graph into a DLC file that can then be cached and plugged into the Android app in [this repository](github.com/tanvikad/qualcomm-hmc-app/edit/master/README.md). The code within this repository is adapted from [QIDK's NLP Solution 1 for Question Answering](https://github.com/quic/qidk/tree/master/Solutions/NLPSolution1-QuestionAnswering). 

There are also a few files for installing and running [AIMET](https://github.com/quic/aimet), which we were able to install and run but did not use in terms of quantization. 

## Steps to generate a DLC file
1.  Generate DistilBERT as a Tensorflow frozen graph:
    
    `python ./scripts/distilbert.py`
    
    After running this command, a model named `distilbert-uncased.pb` will be generated in the `./frozen_models` folder.

    Note: if you want to convert a different model into a frozen graph, replace the `MODEL_NAME` variable with the suitable model name and adjust the inputs/input signature accordingly (e.g., some models only need `input_ids` and `attention_mask` while other models might also require `token_type_ids`).
2.  Make sure you are running the docker container for the SNPE SDK (you can follow the instructions [here](https://github.com/quic/qidk/tree/master/Tools/snpe-docker)). Now, you are able to use tools within the SNPE SDK.
3.  Convert the DistilBERT Tensorflow frozen graph (`distilbert-uncased.pb`) into a DLC file using the following command:
    
    `snpe-tensorflow-to-dlc -i frozen_models/distilbert-uncased.pb -d input_ids 1,384 -d attention_mask 1,384 --out_node Identity --out_node Identity_1 -o dlc_files/distilbert-uncased.dlc`
    
    Note that `input_ids` and `attention_mask` must match the input of the model, and `Identity` and `Identity_1` must match the output of the model. You can load your frozen graph of the model (`distilbert-uncased.pb`) onto Netron to view the names of these.

    Now, there will be a file named `distilbert-uncased.dlc` in the `dlc_files` folder.
4.  Optionally, you can cache the DLC file in order to optimize the model's loading time using DSP runtime (which our app does):
    `snpe-dlc-graph-prepare --input_dlc dlc_files/distilbert-uncased.dlc --use_float_io --htp_archs v73 --set_output_tensors Identity:0,Identity_1:0`

    This generates a DLC file named `distilbert-uncased-cached.dlc` in the `frozen_models` folder.

## Other files
- `scripts/check_onnx.py` is a simple script that checks whether a given ONNX model can be loaded, and prints the input and output names of the model.
- `scripts/aimet.sh` is a bash script for AIMET, which we started using for quantization but did not finish.
- `requirements.txt` lists the requirements for installing and running AIMET. 
