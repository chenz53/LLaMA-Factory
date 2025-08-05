from transformers import AutoConfig, AutoModelForCausalLM
import torch

config = AutoConfig.from_pretrained("standardmodelbio/Qwen3-WM-0.6B", trust_remote_code=True)
print(config)
print(type(config))

model = AutoModelForCausalLM.from_pretrained(
    "standardmodelbio/Qwen3-WM-0.6B",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(model)
