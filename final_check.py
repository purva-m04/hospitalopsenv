import os 
files = ['app/graders.py','app/rewards.py','app/env.py','app/models.py','hospitalopsenv/app/graders.py','hospitalopsenv/app/rewards.py','inference.py'] 
issues = [] 
for path in files: 
    try: 
        lines=open(path,encoding='utf-8').read().splitlines() 
        for i,l in enumerate(lines): 
            if 'return 1.0' in l or 'return 0.0' in l: issues.append(path+':'+str(i+1)+': '+l.strip()) 
    except Exception as e: print('SKIP',path,e) 
print('SCORE ISSUES:',issues if issues else 'NONE') 
