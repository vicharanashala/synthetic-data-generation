import time
from openai import OpenAI

client = OpenAI(base_url="http://100.100.108.100:8080/v1", api_key="dummy")

start = time.time()
resp = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[{"role": "user", "content": "Explain nitrogen fertilizer use in wheat crops in detail."}],
    max_tokens=500,
    temperature=0.75,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
elapsed = time.time() - start

output_tokens = resp.usage.completion_tokens
print(f"Output tokens : {output_tokens}")
print(f"Time taken    : {elapsed:.2f}s")
print(f"Throughput    : {output_tokens / elapsed:.1f} tok/s")