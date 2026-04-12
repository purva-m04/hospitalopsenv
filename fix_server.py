c=open('app_server.py',encoding='utf-8').read() 
c=c.replace('env_copy["USE_HEURISTIC"] = "0"','env_copy["USE_HEURISTIC"] = "1"') 
open('app_server.py','w',encoding='utf-8').write(c) 
print('done') 
