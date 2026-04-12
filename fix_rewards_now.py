import re 
c = open('app/rewards.py').read() 
c = 'def _clamp(v):\n    if v <= 0: return 0.001\n    if v >= 1: return 0.999\n    return round(v, 4)\n\n' + c 
c = c.replace('return self._reward_report(action, outcome, state)', 'return _clamp(self._reward_report(action, outcome, state))') 
c = c.replace('return self._reward_billing(action, outcome, state)', 'return _clamp(self._reward_billing(action, outcome, state))') 
c = c.replace('return self._reward_blood_bank(action, outcome, state)', 'return _clamp(self._reward_blood_bank(action, outcome, state))') 
c = c.replace('return self._reward_icu(action, outcome, state)', 'return _clamp(self._reward_icu(action, outcome, state))') 
open('app/rewards.py','w').write(c) 
print('Done') 
