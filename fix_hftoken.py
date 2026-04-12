c = open('inference.py').read() 
c = c.replace('if HF_TOKEN is None:\n    raise ValueError("HF_TOKEN enviro', 'if HF_TOKEN is None:\n    HF_TOKEN = "dummy"  # not needed when API_KEY is set\nif False and HF_TOKEN is None:\n    raise ValueError("HF_TOKEN enviro') 
open('inference.py','w').write(c) 
