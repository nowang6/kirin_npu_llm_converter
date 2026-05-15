import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.qwen3 import (
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)


def select_providers():
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or available


def main():
    model_path = Path("models", "Qwen3-0.6B")
    onnx_model_path = Path("output/qwen3_0.6b.onnx")

    print("Loading tokenizer...")
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=Path(model_path, "tokenizer.json"),
        repo_id=str(model_path),
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )

    print("Initializing ONNX Runtime session...")
    session = ort.InferenceSession(
        onnx_model_path.as_posix(),
        providers=select_providers(),
    )
    outputs = session.get_outputs()
    output_name = outputs[0].name
    output_names = [out.name for out in outputs]

    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)
    print(f"Input prompt: {prompt}")
    print(f"Input token count: {len(input_token_ids)}")

    torch_context_limit = QWEN_CONFIG_06_B["context_length"]
    actual_context_size = min(len(input_token_ids) + 150, 2048)
    actual_context_size = min(actual_context_size, torch_context_limit)
    print(
        f"Using context size: {actual_context_size} "
        f"(max: {torch_context_limit})"
    )

    max_new_tokens = 150
    eos_id = tokenizer.eos_token_id

    tokens = np.array(input_token_ids, dtype=np.int64)
    n_layers = QWEN_CONFIG_06_B["n_layers"]
    num_kv_groups = QWEN_CONFIG_06_B["n_kv_groups"]
    head_dim = QWEN_CONFIG_06_B["head_dim"]

    def ort_type_to_numpy(dtype_str: str) -> np.dtype:
        if dtype_str == "tensor(float16)":
            return np.float16
        if dtype_str == "tensor(float)":
            return np.float32
        if dtype_str == "tensor(bfloat16)":
            # ORT expects bfloat16 values packed into uint16
            return np.uint16
        raise ValueError(f"Unsupported ONNX input dtype: {dtype_str}")

    input_types = {inp.name: inp.type for inp in session.get_inputs()}
    has_kv_cache = any(name.startswith("past_key_") for name in input_types)

    def run_session(
        token_array: np.ndarray,
        past_key_values: list[np.ndarray] | None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        inputs = {"input_ids": token_array[np.newaxis, :]}
        if has_kv_cache and past_key_values is not None:
            for i in range(n_layers):
                inputs[f"past_key_{i}"] = past_key_values[2 * i]
                inputs[f"past_value_{i}"] = past_key_values[2 * i + 1]
        outputs = session.run(output_names, inputs)
        return outputs[0], outputs[1:] if has_kv_cache else []

    # Initialize empty KV cache tensors for first call (if model supports KV cache)
    empty_cache: list[np.ndarray] | None = None
    if has_kv_cache:
        cache_dtype = ort_type_to_numpy(input_types["past_key_0"])
        empty_cache = []
        for _ in range(n_layers):
            empty_cache.append(
                np.zeros((1, num_kv_groups, 0, head_dim), dtype=cache_dtype)
            )
            empty_cache.append(
                np.zeros((1, num_kv_groups, 0, head_dim), dtype=cache_dtype)
            )

    print("\nStarting ONNX Runtime generation...")
    print("Generated text: ", end="", flush=True)
    start = time.time()

    # Prefill with the full prompt to prime KV cache (or just run once if no cache)
    logits, past_key_values = run_session(tokens, empty_cache)

    for _ in range(max_new_tokens):
        next_token_id = int(np.argmax(logits[0, -1, :]))

        tokens = np.append(tokens, next_token_id)

        if eos_id is not None and next_token_id == eos_id:
            break

        token_text = tokenizer.decode([next_token_id])
        print(token_text, end="", flush=True)

        if has_kv_cache:
            # Feed only the new token with KV cache
            logits, past_key_values = run_session(
                np.array([next_token_id], dtype=np.int64),
                past_key_values,
            )
        else:
            # No KV cache: re-run with full sequence
            logits, past_key_values = run_session(tokens, None)

    elapsed = time.time() - start
    print()
    print(f"Time: {elapsed:.2f} sec")
    if elapsed > 0:
        print(f"{int(len(tokens) / elapsed)} tokens/sec")

    output_text = tokenizer.decode(tokens.tolist())
    print("\n\nOutput text:\n")
    print(output_text + "...")


if __name__ == "__main__":
    main()

