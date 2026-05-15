#!/usr/bin/env bash
set -e

rm -rf omc_out omc.log omc_9020.log onnx_embedding_out_no_output_pos dopt_out
source .venv/bin/activate
python src/opt_main.py --quant-stage stage1
python src/opt_main.py --quant-stage stage2
python src/opt_main.py --quant-stage stage3
python src/export_model_single_qwen2.py
./omc_sh/qwen25_0.5b.sh >> omc_9020.log
