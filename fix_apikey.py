import re 
c = open('inference.py').read() 
old = 'client = OpenAI(api_key=os.environ.get("API_KEY", HF_TOKEN), base_url=API_BASE_URL)' 
new = 'api_key_val = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or HF_TOKEN' 
new += chr(10) + '        client = OpenAI(api_key=api_key_val, base_url=API_BASE_URL)' 
c = c.replace(old, new) 
open('inference.py','w').write(c) 
print('Done:', 'api_key_val' in open('inference.py').read()) 
