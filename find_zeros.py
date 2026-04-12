import re 
c=open('inference.py').read() 
matches=[c[m.start()-50:m.end()+50] for m in re.finditer(r'grader_score', c)] 
[print(m) for m in matches] 
