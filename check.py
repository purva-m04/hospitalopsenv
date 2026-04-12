content = open('inference.py', encoding='utf-8').read() 
errors = [] 
if 'grader_score = 0.0' in content: errors.append('FAIL: grader_score = 0.0 still exists') 
if 'or 0.0)' in content: errors.append('FAIL: or 0.0 still exists') 
if 'max(0.001, min(0.999' not in content: errors.append('FAIL: clamping missing') 
print('ALL GOOD - safe to resubmit!' if not errors else '\n'.join(errors)) 
