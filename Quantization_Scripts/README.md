## Description
This repository contains the code necessary to export an off-the-shelf machine learning language model as a DLC file, which is then plugged into the 2023-24 Qualcomm CS Clinic team's Android app project. This README file contains the steps necessary to obtain a transformers-based language model (specifically DistilBERT, but the same process can be applied to other models of similar architectures), convert it into a Tensorflow frozen graph, and convert the frozen graph into a DLC file that can then be cached and plugged into the Android app in [this repository](github.com/tanvikad/qualcomm-hmc-app/edit/master/README.md). The code within this repository is adapted from [QIDK's NLP Solution 1 for Question Answering](https://github.com/quic/qidk/tree/master/Solutions/NLPSolution1-QuestionAnswering). 

## Steps to generate a DLC file
1.  Generate DistilBERT as a Tensorflow frozen graph:
    
    `python3 ./models_to_export/distilbert.py`
    
    After running this command, a model named `distilbert-uncased.pb` will be generated in the `./frozen_models` folder.
2.  Make sure you are running the docker container for the SNPE SDK (you can follow the instructions [here](https://github.com/quic/qidk/tree/master/Tools/snpe-docker)). Now, you are able to use tools within the SNPE SDK.
3.  Convert the DistilBERT Tensorflow frozen graph (`distilbert-uncased.pb`) into a DLC file using the following command:
    
    `snpe-tensorflow-to-dlc -i frozen_models/distilbert-uncased.pb -d input_ids 1,384 -d attention_mask 1,384 --out_node Identity --out_node Identity_1 -o dlc_files/disilbert-uncased.dlc`
    
    Note that `input_ids` and `attention_mask` must match the input of the model, and `Identity` and `Identity_1` must match the output of the model. You can load your frozen graph of the model (`distilbert-uncased.pb`) onto Netron to view the names of these.

    Now, there will be a file named `distilbert-uncased.dlc` in the `dlc_files` folder.
4.  Optionally, you can cache the DLC file in order to optimize the model's loading time using DSP runtime (which our app does):
    `snpe-dlc-graph-prepare --input_dlc dlc_files/distilbert-uncased.dlc --use_float_io --htp_archs v73 --set_output_tensors Identity:0,Identity_1:0`

    This generates a DLC file named `distilbert-uncased-cached.dlc` in the `frozen_models` folder.

## Other files
- `models_to_export/check_onnx.py` is a simple script that checks whether a given ONNX model can be loaded, and prints the input and output names of the model.
- 