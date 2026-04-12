import re 
c = open('inference.py').read() 
c = c.replace('    api_key_val = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or HF_TOKEN\n        client = OpenAI(api_key=api_key_val, base_url=API_BASE_URL)', '        api_key_val = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or HF_TOKEN\n        client = OpenAI(api_key=api_key_val, base_url=API_BASE_URL)') 
open('inference.py','w').write(c) 
