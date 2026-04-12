c=open('inference.py',encoding='utf-8').read() 
c=c.replace('"bloodbank_easy", "bloodbank_medium", "bloodbank_hard",','"bloodbank_easy", "bloodbank_medium", "bloodbank_hard",\n    "icu_easy",      "icu_medium",      "icu_hard",') 
open('inference.py','w',encoding='utf-8').write(c) 
print('done') 
