scores=[0.80+0.20,0.10+0.15+0.20+0.20+0.20+0.35,0.35+0.15+0.20+0.15+0.15,0.15+0.15+0.15+0.15+0.25+0.20+0.10] 
for i,s in enumerate(scores): print('Task',i+1,'max:',round(s,3),'clamped:',min(0.999,max(0.001,round(s,3)))) 
