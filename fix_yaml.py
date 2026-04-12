c=open('openenv.yaml',encoding='utf-8').read() 
c=c.replace('api_key: OPENAI_API_KEY','api_key: HF_TOKEN') 
open('openenv.yaml','w',encoding='utf-8').write(c) 
print('done') 
