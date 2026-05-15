from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_glm import GlmForCausalLM

MODEL_PATH = "/home/ma-user/workspace/Kirin_AI_User/AIC/y00838596/ckpt/zhipu-GLM-1.5B-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = GlmForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
print(tokenizer)

message = [{"role": "user", "content": "who are you?"}]

inputs = tokenizer.apply_chat_template(
    message,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

generate_kwargs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "max_new_tokens": 128,
    "do_sample": False,
}
print(inputs["input_ids"])
out = model.generate(**generate_kwargs)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
