from app.env import HospitalOpsEnv 
import json 
env = HospitalOpsEnv('scenarios') 
obs = env.reset('bloodbank_hard') 
print(json.dumps(obs.task_context, indent=2)) 
