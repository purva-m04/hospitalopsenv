issues=[(i+1,l) for i,l in enumerate(open('hf_check.py',encoding='utf-8').read().splitlines()) if 'grader_score' in l and '0.0' in l and '0.001' not in l] 
print('HF PROBLEMS:',issues) if issues else print('HF ALL CLEAN!') 
