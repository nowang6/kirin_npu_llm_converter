from src.ch05 import generate_one_token
import time
from pathlib import Path
import torch
from safetensors.torch import load_file

from src.qwen3 import Qwen3Model, QWEN_CONFIG_06_B, load_weights_into_qwen, Qwen3Tokenizer

model_path= "models/Qwen3-0.6B"

model_file = Path(model_path,"model.safetensors")

print("Loading model...")
model = Qwen3Model(QWEN_CONFIG_06_B)
weights_dict = load_file(model_file)
load_weights_into_qwen(model, QWEN_CONFIG_06_B, weights_dict)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"Using device: {device}")
model.to(device);
model.eval()  # Set to evaluation mode


USE_INSTRUCT_MODEL = False
USE_REASONING_MODEL = True


if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

print("Loading tokenizer...")
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=Path(model_path,"tokenizer.json"),
    repo_id=model_path,
    apply_chat_template=USE_REASONING_MODEL,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=not USE_INSTRUCT_MODEL
)

prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
print(f"Input prompt: {prompt}")
print(f"Input token count: {len(input_token_ids)}")

torch.manual_seed(123)

# Use a more reasonable context size - use actual input length + some buffer
# Instead of the full 40960, which is too large for CPU
actual_context_size = min(len(input_token_ids) + 150, 2048)  # Reasonable size for CPU
print(f"Using context size: {actual_context_size} (max: {QWEN_CONFIG_06_B['context_length']})")

start = time.time()
print("\nStarting generation...")
print("Generated text: ", end="", flush=True)

# Initialize with input tokens
idx = torch.tensor(input_token_ids, device=device).unsqueeze(0)
max_new_tokens = 5
eos_id = None  # Can be set to stop token ID if needed

# Generate tokens one by one
for i in range(max_new_tokens):
    next_token = generate_one_token(
        model=model,
        idx=idx,
        context_size=actual_context_size,
        top_k=1,
        temperature=0.,
        eos_id=eos_id
    )
    
    # Check for EOS token if specified
    if eos_id is not None and next_token.item() == eos_id:
        break
    
    # Decode and print the new token immediately
    token_text = tokenizer.decode([next_token.item()])
    print(token_text, end="", flush=True)
    
    # Append the new token to the sequence
    idx = torch.cat((idx, next_token), dim=1)

print()  # New line after generation

output_token_ids = idx

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
