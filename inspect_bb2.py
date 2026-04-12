from app.env import HospitalOpsEnv 
from app.models import Action, ActionType 
import json 
env = HospitalOpsEnv('scenarios') 
obs = env.reset('bloodbank_hard') 
a = Action(action_type=ActionType('discard_expired'), payload={'blood_type':'A+','units_to_discard':3}, episode_id=obs.episode_id) 
obs2,r,d,info = env.step(a) 
print(json.dumps(obs2.task_context, indent=2)) 
