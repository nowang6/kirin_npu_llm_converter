# 目录介绍

## 主要目录结构

- **`src/`** - 源代码目录
  - `opt_main.py` - 量化主程序
  - `export_model_single_qwen2.py` - Qwen2模型导出脚本
  - `qwen_test.py` - 量化测试脚本
  - `utils.py` - 工具函数

- **`models/`** - 模型文件目录
  - 存放不同版本的Qwen模型（Qwen2.5-0.5B、Qwen2.5-1.5B、Qwen3-0.6B等）

- **`dopt_conf/`** - 量化配置文件目录
  - 包含各模型的量化配置文件（JSON格式）

- **`dopt_out/`** - 量化输出目录
  - 存放量化过程中生成的权重文件、配置文件等

- **`npu_tuned_model/`** - NPU调优模型代码
  - `qwen2/` - Qwen2模型相关实现
  - `qwen3/` - Qwen3模型相关实现
  - `glm/` - GLM模型相关实现

- **`omc_sh/`** - OMC转换脚本目录
  - 包含将ONNX模型转换为OM模型的shell脚本

- **`omc_out/`** - OMC转换输出目录
  - 存放转换后的OM模型文件

- **`onnx_out_embedding_out_no_output_pos/`** - ONNX模型输出目录
  - 存放导出的ONNX模型文件

- **`qwen2_final/`** - Qwen2最终部署文件目录
  - 包含部署所需的权重文件和配置文件


- **`tools/`** - 工具目录
  - `tools_dopt/` - 量化工具（PyTorch、ONNX、TensorFlow）
  - `tools_omg/` - OMG模型转换工具
  - `tools_ascendc/` - AscendC工具链
  - `platform/` - 平台相关配置和库

- **`test/`** - 测试目录
  - 包含各种测试脚本和配置文件

- **`logs/`** - 日志目录
  - 存放运行过程中生成的日志文件

- **`model_info_target/`** - 模型信息目标配置目录
  - 包含各模型的YAML配置文件

# 环境
```sh
source ./venv/bin/activate
PYTHONPATH=.:./tools/tools_dopt/dopt_pytorch_py3
```


# 量化
## 权重量化
```sh
python src/opt_main.py --quant-stage stage1
```
## 激活量化
```sh
python src/opt_main.py --quant-stage stage2
```
## 参数提取
```sh
python src/opt_main.py --quant-stage stage3
```

##量化测试
```sh
python src/qwen_test.py
```

# 转换为onnx

```sh
python src/export_model_single_qwen2.py
```

# ONNX推理

导出ONNX模型后，可以使用ONNX Runtime进行推理：

```sh
python src/qwen_onnx_test.py --prompt "who are you?" --max-new-tokens 50
```

更多参数：
- `--temperature`: 采样温度（默认1.0）
- `--top-p`: top-p采样参数（默认1.0）
- `--do-sample`: 启用采样
- `--chat`: 使用聊天格式

# 转换为om
```sh
omc_sh/qwen25_1.5b.sh
```

# 部署

qwen2_final/executor.json


# DataType枚举定义

```cpp
enum DataType {
    DT_UNDEFINED = 17, // Used to indicate a DataType field has not been set.
    DT_FLOAT = 0, // float type
    DT_FLOAT16 = 1, // fp16 type
    DT_INT8 = 2, // int8 type
    DT_INT16 = 6, // int16 type
    DT_UINT16 = 7, // uint16 type
    DT_UINT8 = 4, // uint8 type
    DT_INT32 = 3,
    DT_INT64 = 9, // int64 type
    DT_UINT32 = 8, // unsigned int32
    DT_UINT64 = 10, // unsigned int64
    DT_BOOL = 12, // bool type
    DT_DOUBLE = 11, // double type
    DT_DUAL = 13, /* dual output type */
    DT_DUAL_SUB_INT8 = 14, /* dual output int8 type */
    DT_DUAL_SUB_UINT8 = 15, /* dual output uint8 type */
    DT_COMPLEX64 = 16,
    DT_2BIT = 21, // 2BIT type
    DT_INT4 = 22, // int4 type
    DT_QUINT8 = 23, // quint8 for ann
    DT_RESOURCE = 24, // dataflow resource flow
    DT_3BIT = 25, // 3BIT type
    DT_UINT2 = 26, // uint2 type
    DT_UINT4 = 27, // uint4 type
    DT_STRING = 28, // string type
    DT_FLOAT8_E5M2 = 35,
    DT_FLOAT4_E2M1 = 40,
    DT_MAX
};
```