#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025-2035. All rights reserved
import os
import argparse
from dopt.log import Logger
from dopt.dopt_lm.train import generate_config
from dopt.dopt_lm.train import inout_quant, weight_quant_pipline
from dopt.dopt_lm.train import build_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str, default="/data/models/Qwen2.5-1.5B-Instruct", help="model name"
    )
    parser.add_argument(
        "--dopt-config", type=str, default="dopt_conf/qwen2.5-1.5b.json"
    )
    parser.add_argument(
        "--optimize-config", type=str, default="optimize-config.yaml"
    )
    parser.add_argument(
        "--group-size", type=int, default=128
    )
    parser.add_argument(
        "--block-size", type=int, default=128
    )
    parser.add_argument(
        "--act-bits", type=int, default=16
    )
    parser.add_argument(
        "--w-bits", type=int, default=4
    )
    parser.add_argument(
        "--output-dir",
        type=str, default="dopt_out", help="output_dir"
    )
    parser.add_argument("--hf_type_save", action='store_true', default=False)
    parser.add_argument("--quant-stage", type=str, default='stage3', help="quant stage")
    args = parser.parse_args()
    return args


def generate_quant_config(model_name_or_path, config_path):
    generate_config(model_name_or_path, config_path)


def llm_quant_pipline(**argv):
    os.environ["custom_group_size"] = str(argv.get('group_size'))
    os.environ["custom_block_size"] = str(argv.get('block_size'))
    os.environ["custom_act_bits"] = str(argv.get('act_bits'))
    os.environ["custom_w_bits"] = str(argv.get('w_bits'))
    if argv.get("quant_stage", 'stage1') == "stage1":
        weight_quant_pipline(**argv)
    elif argv.get("quant_stage", 'stage1') == "stage2":
        inout_quant(**argv)
    elif argv.get("quant_stage", 'stage1') == "stage3":
        build_params(**argv)
    else:
        raise Exception(f'quant stage1 or stage2 is valid')


if __name__ == '__main__':
    args = parse_args()
    ### step1 generate dopt_quant_config
    if not os.path.exists(args.dopt_config):
        generate_quant_config(args.model_path, args.dopt_config)
        Logger.info(f'generate plugin quang config please set quant strategy firstly')
        exit()
    optimize_info = {
        "model_path" :  args.model_path,
        "config_yaml":  args.optimize_config,
        "dopt_config":  args.dopt_config,
        "output_dir" :  args.output_dir,
        "hf_type_save": args.hf_type_save,
        "quant_stage" : args.quant_stage,
        "group_size" :  args.group_size,
        "block_size" :  args.block_size,
        "act_bits":     args.act_bits,
        "w_bits":       args.w_bits,
    }
    ### step2 run llm model Quantification optimization process
    llm_quant_pipline(**optimize_info)
