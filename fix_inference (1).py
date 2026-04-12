c = open('inference.py', 'rb').read().decode('utf-8')

c = c.replace('grader_score = 0.0', 'grader_score = 0.001')
c = c.replace('info.grader_score or 0.0', 'max(0.001, min(0.999, info.grader_score or 0.001))')

# Add clamping line before the return statement in run_episode
old = '    return {\n        "scenario_id": scenario_id,'
new = '    grader_score = max(0.001, min(0.999, grader_score))\n    return {\n        "scenario_id": scenario_id,'
if old in c:
    c = c.replace(old, new)
    print('Clamping added (unix line endings)')
else:
    old = '    return {\r\n        "scenario_id": scenario_id,'
    new = '    grader_score = max(0.001, min(0.999, grader_score))\r\n    return {\r\n        "scenario_id": scenario_id,'
    if old in c:
        c = c.replace(old, new)
        print('Clamping added (windows line endings)')
    else:
        print('WARNING: could not find return block - searching...')
        idx = c.find('"scenario_id": scenario_id,')
        print(f'Found at index {idx}, surrounding text:')
        print(repr(c[idx-50:idx+50]))

open('inference.py', 'wb').write(c.encode('utf-8'))
print('DONE - inference.py fixed!')
