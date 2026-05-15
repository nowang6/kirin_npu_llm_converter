#!/usr/bin/env python3
"""
基于ONNX模型的Qwen2.5-0.5B文本生成器（固定长度输入版本）。
模型被导出为固定序列长度64，需要特殊处理填充和掩码。
"""

import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

class QwenONNXGeneratorFixed:
    def __init__(
        self,
        onnx_model_path: str,
        embed_dir: str,
        tokenizer_path: str = "models/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ):
        """
        初始化ONNX文本生成器（固定长度版本）。

        参数:
            onnx_model_path: ONNX模型路径
            embed_dir: 嵌入权重目录
            tokenizer_path: 分词器路径
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: top-p采样参数
            do_sample: 是否采样（False时为贪婪解码）
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.fixed_seq_len = 64  # 模型导出的固定序列长度

        # 加载分词器
        print("加载分词器...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # 加载ONNX模型
        print(f"加载ONNX模型: {onnx_model_path}")
        self.session = ort.InferenceSession(onnx_model_path)

        # 模型配置信息
        self.hidden_size = 896
        self.num_layers = 24
        self.num_attention_heads = 14
        self.num_key_value_heads = 2
        self.head_dim = self.hidden_size // self.num_attention_heads

        # 检查KV缓存最大长度
        first_past_key = next(inp for inp in self.session.get_inputs()
                              if inp.name.startswith("past_key_in"))
        self.kv_cache_max_len = first_past_key.shape[0]  # 2048

        print(f"模型配置: hidden_size={self.hidden_size}, layers={self.num_layers}")
        print(f"KV缓存最大长度: {self.kv_cache_max_len}")
        print(f"固定序列长度: {self.fixed_seq_len}")

        # 加载嵌入权重和缩放因子
        print("加载嵌入权重和缩放因子...")
        self.embed_weights, self.embed_scales = self._load_embedding_data(
            embed_dir, seq_len=self.fixed_seq_len, kv_cache_max_len=self.kv_cache_max_len
        )
        self.vocab_size = self.embed_weights.shape[0]

        # 初始化KV缓存和状态
        self._init_kv_cache()
        self.kv_cache_pos = 0  # 当前KV缓存中的有效token数

    def _load_embedding_data(self, embed_dir: str, seq_len: int, kv_cache_max_len: int):
        """加载嵌入权重和缩放因子"""
        embed_dir = Path(embed_dir)
        weight_path = embed_dir / f"model_{seq_len}_{kv_cache_max_len}.embedding_weights"
        scale_path = embed_dir / f"model_{seq_len}_{kv_cache_max_len}.embedding_dequant_scale"

        # 嵌入权重: int8
        vocab_size = 151936
        hidden_size = 896
        weights = np.fromfile(weight_path, dtype=np.int8).reshape(vocab_size, hidden_size)

        # 缩放因子: float32
        scales = np.fromfile(scale_path, dtype=np.float32)
        if scales.shape[0] != vocab_size:
            print(f"警告: 缩放因子形状 {scales.shape} 与词汇表大小 {vocab_size} 不匹配")
            # 尝试调整形状
            if scales.shape[0] == vocab_size * hidden_size:
                scales = scales.reshape(vocab_size, hidden_size)
            else:
                raise ValueError(f"无法识别缩放因子形状: {scales.shape}")
        else:
            scales = scales.reshape(vocab_size, 1)

        return weights, scales

    def _init_kv_cache(self):
        """初始化KV缓存为零矩阵"""
        batch = 1
        self.kv_cache_shape = (
            self.kv_cache_max_len,
            self.num_key_value_heads,
            batch,
            self.head_dim,
        )
        self.kv_cache = []
        for _ in range(self.num_layers):
            past_key = np.zeros(self.kv_cache_shape, dtype=np.float32)
            past_value = np.zeros(self.kv_cache_shape, dtype=np.float32)
            self.kv_cache.append((past_key, past_value))

    def _prepare_inputs(
        self, input_ids: np.ndarray, effective_len: int
    ) -> Dict[str, np.ndarray]:
        """
        准备模型输入。

        参数:
            input_ids: 填充后的token IDs，形状[1, fixed_seq_len]
            effective_len: 实际有效token数（非填充部分长度）

        返回:
            输入字典
        """
        batch, seq_len = input_ids.shape
        assert seq_len == self.fixed_seq_len, f"输入长度必须是{self.fixed_seq_len}，实际是{seq_len}"

        # input_embed
        input_embed = self.embed_weights[input_ids].astype(np.int8)  # [1, 64, 896]

        # embed_scales
        embed_scales = self.embed_scales[input_ids]  # [1, 64, 1] 或 [1, 64, 896]
        if embed_scales.ndim == 2:
            embed_scales = embed_scales.reshape(batch, seq_len, 1)

        # attention_mask
        attention_mask = np.zeros((batch, 1, seq_len, self.kv_cache_max_len), dtype=np.float32)

        # 对于每个有效token（查询位置），它可以看见所有之前的有效token
        for i in range(effective_len):
            # 可见长度 = 当前KV缓存位置 + i + 1
            visible_len = self.kv_cache_pos + i + 1
            if visible_len > self.kv_cache_max_len:
                visible_len = self.kv_cache_max_len
            attention_mask[:, :, i, :visible_len] = 1.0

        # 填充token位置：完全掩码（看不到任何token）
        for i in range(effective_len, seq_len):
            attention_mask[:, :, i, :] = 0.0

        # position_ids
        position_ids = np.zeros((batch, seq_len), dtype=np.int32)
        for i in range(effective_len):
            position_ids[0, i] = self.kv_cache_pos + i

        # new_kv_cache_pos: 长度为64，有效token写入递增位置，填充token写入-1
        new_kv_cache_pos = np.full((seq_len,), -1, dtype=np.int32)
        for i in range(effective_len):
            new_kv_cache_pos[i] = self.kv_cache_pos + i

        # 构建输入字典
        inputs = {}
        inputs["input_embed"] = input_embed
        inputs["attention_mask"] = attention_mask
        inputs["position_ids"] = position_ids
        inputs["new_kv_cache_pos"] = new_kv_cache_pos
        inputs["embed_scales"] = embed_scales

        for i in range(self.num_layers):
            inputs[f"past_key_in{i}"] = self.kv_cache[i][0]
            inputs[f"past_value_in{i}"] = self.kv_cache[i][1]

        return inputs

    def _update_kv_cache(self, outputs: List[np.ndarray]):
        """从模型输出更新KV缓存"""
        for i in range(self.num_layers):
            key_idx = 1 + i * 2
            value_idx = key_idx + 1
            self.kv_cache[i] = (outputs[key_idx], outputs[value_idx])

    def _get_next_token(self, logits: np.ndarray, effective_len: int) -> int:
        """
        根据logits选择下一个token。

        参数:
            logits: 形状[1, fixed_seq_len, vocab_size]
            effective_len: 实际有效token数

        返回:
            下一个token ID
        """
        # 取最后一个有效token位置的logits
        last_token_logits = logits[0, effective_len - 1, :]  # [vocab_size]

        if self.do_sample:
            # 采样模式
            if self.temperature > 0:
                last_token_logits = last_token_logits / self.temperature

            # top-p采样
            if self.top_p < 1.0:
                sorted_indices = np.argsort(last_token_logits)[::-1]
                sorted_logits = last_token_logits[sorted_indices]
                cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))

                # 移除累积概率大于top_p的token
                indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove = sorted_indices[indices_to_remove]
                last_token_logits[sorted_indices_to_remove] = -float("inf")

            # softmax和采样
            probs = np.exp(last_token_logits) / np.sum(np.exp(last_token_logits))
            next_token_id = np.random.choice(self.vocab_size, p=probs)
        else:
            # 贪婪解码
            next_token_id = np.argmax(last_token_logits)

        return int(next_token_id)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """
        生成文本。

        参数:
            prompt: 输入提示文本
            max_new_tokens: 最大生成token数（覆盖初始化值）
            temperature: 采样温度（覆盖初始化值）
            top_p: top-p采样参数（覆盖初始化值）
            do_sample: 是否采样（覆盖初始化值）

        返回:
            生成的文本
        """
        # 使用参数或默认值
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        do_sample = do_sample or self.do_sample

        # 编码输入 - 使用聊天模板（与PyTorch版本保持一致）
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=self.fixed_seq_len,  # 最多固定长度
        )
        input_ids_original = inputs["input_ids"].astype(np.int64)  # [1, original_len]
        original_len = input_ids_original.shape[1]

        print(f"输入: '{prompt}'")
        print(f"处理后输入长度: {original_len}")

        # 重置状态
        self._init_kv_cache()
        self.kv_cache_pos = 0

        # 处理初始输入
        generated_ids = []
        all_input_ids = []

        # 如果输入长度超过固定长度，需要分块处理
        if original_len > self.fixed_seq_len:
            print(f"警告: 输入长度{original_len}超过固定长度{self.fixed_seq_len}，将截断")
            input_ids_original = input_ids_original[:, :self.fixed_seq_len]
            original_len = self.fixed_seq_len

        # 填充到固定长度
        if original_len < self.fixed_seq_len:
            # 填充
            pad_len = self.fixed_seq_len - original_len
            padding = np.full((1, pad_len), self.pad_token_id, dtype=np.int64)
            input_ids = np.concatenate([input_ids_original, padding], axis=1)
        else:
            input_ids = input_ids_original

        print(f"填充后输入形状: {input_ids.shape}")

        # 第一次推理：处理所有初始token
        print("处理初始输入...")
        inputs_dict = self._prepare_inputs(input_ids, effective_len=original_len)

        output_names = [out.name for out in self.session.get_outputs()]
        outputs = self.session.run(output_names, inputs_dict)

        # 更新KV缓存和位置
        self._update_kv_cache(outputs)
        self.kv_cache_pos += original_len

        # 获取logits并选择第一个生成token
        lm_logits = outputs[0]  # [1, 64, vocab_size]
        next_token_id = self._get_next_token(lm_logits, effective_len=original_len)
        generated_ids.append(next_token_id)
        all_input_ids = list(input_ids_original[0]) + [next_token_id]

        print(f"生成token 1/{max_new_tokens}: {next_token_id}")

        # 自回归生成循环
        for step in range(1, max_new_tokens):
            # 构建固定长度输入：新token在第一个位置，其余为填充
            current_input = np.full((1, self.fixed_seq_len), self.pad_token_id, dtype=np.int64)
            current_input[0, 0] = next_token_id  # 新token在位置0

            inputs_dict = self._prepare_inputs(current_input, effective_len=1)

            # 运行推理
            outputs = self.session.run(output_names, inputs_dict)

            # 更新KV缓存和位置
            self._update_kv_cache(outputs)
            self.kv_cache_pos += 1

            # 获取下一个token
            lm_logits = outputs[0]
            next_token_id = self._get_next_token(lm_logits, effective_len=1)
            generated_ids.append(next_token_id)
            all_input_ids.append(next_token_id)

            print(f"生成token {step+1}/{max_new_tokens}: {next_token_id}")

            # 检查是否遇到终止符
            if next_token_id == self.tokenizer.eos_token_id:
                print(f"遇到终止符，停止生成")
                break

            # 检查是否达到最大长度
            if self.kv_cache_pos >= self.kv_cache_max_len:
                print(f"达到KV缓存最大长度，停止生成")
                break

        # 解码生成的文本
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return generated_text

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        聊天格式生成。

        参数:
            messages: 消息列表，每个消息包含role和content
            kwargs: 传递给generate的参数

        返回:
            生成的回复
        """
        # 构建Qwen聊天模板
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(text, **kwargs)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2.5 ONNX文本生成（固定长度）")
    parser.add_argument("--prompt", type=str, default="who are you?", help="输入提示")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p采样参数")
    parser.add_argument("--do-sample", action="store_true", help="使用采样（否则贪婪解码）")
    parser.add_argument("--chat", action="store_true", help="使用聊天模式")

    args = parser.parse_args()

    # 初始化生成器
    generator = QwenONNXGeneratorFixed(
        onnx_model_path="onnx_out_embedding_out_no_output_pos/model.onnx",
        embed_dir="onnx_out_embedding_out_no_output_pos",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    if args.chat:
        # 聊天模式
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": args.prompt}
        ]
        print(f"用户: {args.prompt}")
        response = generator.chat(messages)
        print(f"\n助手: {response}")
    else:
        # 普通生成模式
        print(f"提示: {args.prompt}")
        start_time = time.time()
        response = generator.generate(args.prompt)
        elapsed = time.time() - start_time

        print(f"\n生成结果: {response}")
        print(f"\n生成时间: {elapsed:.2f}秒")

    return 0


if __name__ == "__main__":
    sys.exit(main())