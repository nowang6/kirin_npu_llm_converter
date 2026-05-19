mkdir out
export MODEL_NAME=Qwen2.5-1.5B-Instruct
export MODEL_BASE_PATH=.
cp onnx_embedding_out_no_output_pos/model_64_2048.embedding_dequant_scale out/
cp onnx_embedding_out_no_output_pos/model_64_2048.embedding_weights out/
cp omc_out/* out/
cp ${MODEL_BASE_PATH}/${MODEL_NAME}/tokenizer.json out/
