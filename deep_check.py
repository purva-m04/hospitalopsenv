import os 
problems = [] 
skip = ['.venv','venv','__pycache__','fix_','check_','git_','hf_','old.','scan','final','write','inference_new'] 
for root,dirs,files in os.walk('.'): 
    dirs[:] = [d for d in dirs if not any(s in d for s in ['.venv','venv','__pycache__'])] 
    for f in files: 
        if not f.endswith('.py'): continue 
        if any(s in f for s in skip): continue 
        path=os.path.join(root,f) 
        try: 
            lines=open(path,encoding='utf-8').read().splitlines() 
            for i,l in enumerate(lines): 
                stripped=l.strip() 
                if stripped in ['return 0.0','return 1.0','return 0','return 1']: problems.append(path+':'+str(i+1)+': '+stripped) 
        except: pass 
print('EXACT 0/1 RETURNS:',problems if problems else 'NONE FOUND - CLEAN') 
