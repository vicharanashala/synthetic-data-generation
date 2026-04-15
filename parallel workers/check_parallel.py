import time, threading
from openai import OpenAI

client = OpenAI(base_url="http://100.100.108.100:8080/v1", api_key="dummy")

PROMPT = """You are an expert in Indian agricultural Q&A dataset creation.
Generate a paraphrase of this Q&A in formal style.

SEED QUESTION: My wheat crop has yellowing leaves, what should I do?
SEED ANSWER: Yellowing in wheat can be due to nitrogen deficiency. Apply urea at 
50 kg/ha as top dressing. Ensure proper irrigation after application. If symptoms 
persist, spray 2% DAP solution as foliar spray. Avoid waterlogging as it can worsen 
the condition.

Output ONLY valid JSON: {"generated_question": "...", "generated_answer": "..."}"""

def single_call(label, results, idx):
    t0 = time.time()
    client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B",
        messages=[
            {"role": "system", "content": "You are an agricultural AI assistant."},
            {"role": "user", "content": PROMPT}
        ],
        max_tokens=1200,
        temperature=0.75,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    results[idx] = time.time() - t0
    print(f"  {label} done in {results[idx]:.1f}s")

print("Final ceiling test\n")
print(f"  {'Workers':<10} {'Wall time':<12} {'Calls/sec':<12} {'Speedup':<12} {'Status'}")
print(f"  {'-'*65}")

baseline_cps = 0.43

for n_workers in [20, 24, 28, 32, 40]:
    results = [0.0] * n_workers
    threads = [
        threading.Thread(target=single_call, args=(f"w{i}", results, i))
        for i in range(n_workers)
    ]

    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0

    calls_sec = n_workers / wall
    speedup   = calls_sec / baseline_cps

    if wall > 8.0:
        flag = "❌ HARD STOP — GPU saturated"
    elif wall > 6.0:
        flag = "⚠️  ceiling approaching"
    elif wall > 4.5:
        flag = "〰️  slowing down"
    else:
        flag = "✅ still good"

    print(f"  {n_workers:<10} {wall:<12.1f} {calls_sec:<12.2f} {speedup:.2f}x  {flag}")
    print()

print("Recommended N_WORKERS = last ✅ before flag appears")