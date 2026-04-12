lines = open('inference.py').readlines() 
lines[303] = '            api_key_val = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or HF_TOKEN\n' 
lines[304] = '            client = OpenAI(api_key=api_key_val, base_url=API_BASE_URL)\n' 
open('inference.py','w').writelines(lines) 
