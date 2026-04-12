issues=[(i+1,l) for i,l in enumerate(open('github_check.py',encoding='utf-8').read().splitlines()) if 'grader_score' in l and '0.0' in l and '0.001' not in l] 
print('GitHub PROBLEMS:',issues) if issues else print('GitHub ALL CLEAN!') 
