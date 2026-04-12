import os 
patterns = ['return 0.0','return 1.0',' = 1.0',' = 0.0','score = 0','score = 1'] 
files = ['app/graders.py','app/rewards.py','app/env.py','hospitalopsenv/app/graders.py','hospitalopsenv/app/rewards.py','hospitalopsenv/app/env.py','inference.py'] 
for path in files: 
    try: 
        lines=open(path,encoding='utf-8').read().splitlines() 
        for i,l in enumerate(lines): 
            if any(x in l for x in patterns): 
                print(path+':'+str(i+1)+': '+l.strip()) 
    except Exception as e: print('SKIP',path,e) 
