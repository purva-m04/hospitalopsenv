import re 
c=open('app/graders.py').read() 
matches=[c[m.start()-30:m.end()+30] for m in re.finditer(r'return 0\.0', c)] 
print(matches) 
