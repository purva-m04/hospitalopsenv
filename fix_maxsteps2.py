c=open('inference.py',encoding='utf-8').read() 
c=c.replace('MAX_STEPS    = 10','MAX_STEPS    = 15') 
open('inference.py','w',encoding='utf-8').write(c) 
print('done') 
