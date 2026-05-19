#!/usr/bin/env bash
set -e

rm -rf onnx_embedding_out_no_output_pos dopt_out out

source .venv/bin/activate
python src/opt_main.py --quant-stage stage1
python src/opt_main.py --quant-stage stage2
python src/opt_main.py --quant-stage stage3
python src/export_model_single_qwen2.py
./omc_sh/qwen25_1.5b.sh



mkdir out
cp onnx_embedding_out_no_output_pos/model_64_2048.embedding_dequant_scale out/
cp onnx_embedding_out_no_output_pos/model_64_2048.embedding_weights out/
cp omc_out/* out/
cp app_conf/* out/
cp /data/models/Qwen2.5-1.5B-Instruct/tokenizer.json out/