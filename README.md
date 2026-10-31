curl -X POST http://localhost:2034/start_profile


curl http://127.0.0.1:2034/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "messages": [
      {"role": "user", "content": "你好，请帮我解释vllm的定义"}
    ],
    "max_tokens": 10,
    "temperature": 0.7
  }'


curl -X POST http://localhost:2034/stop_profile

from torch_npu.profiler.profiler import analyse
analyse("./profilling")
