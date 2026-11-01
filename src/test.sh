`curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "messages": [{"role": "user", "content": "Hello! What is your model architecture?"}],
    "max_tokens": 64
  }'`

  
