c = open('inference.py').read() 
old = 'print(f"[INFO] LLM agent: {MODEL_NAME} @ {API_BASE_URL}", flush=True)' 
new = 'print(f"[INFO] LLM agent: {MODEL_NAME} @ {API_BASE_URL}", flush=True)\n            try:\n                client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user","content":"ping"}], max_tokens=1, timeout=10)\n            except Exception:\n                pass' 
c = c.replace(old, new) 
open('inference.py','w').write(c) 
print('Done') 
