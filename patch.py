data = open('inference.py', encoding='utf-8', errors='replace').read() 
data = data.replace('\u2550' * 60, '-' * 60) 
open('inference.py', 'w', encoding='utf-8').write(data) 
