c=open('app/graders.py').read() 
c=c.replace('return 0.0','return 0.001') 
open('app/graders.py','w').write(c) 
