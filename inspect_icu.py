from app.env import HospitalOpsEnv 
import json 
env = HospitalOpsEnv('scenarios') 
obs = env.reset('icu_hard') 
print(json.dumps(obs.task_context, indent=2)) 
