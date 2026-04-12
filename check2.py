lines=open('inference.py',encoding='utf-8').read().splitlines() 
issues=[(i+1,l) for i,l in enumerate(lines) if 'grader_score' in l and '0.0' in l and '0.001' not in l] 
print('PROBLEMS:',issues) if issues else print('ALL CLEAN - safe to push!') 
