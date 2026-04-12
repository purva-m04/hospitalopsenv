from app.env import HospitalOpsEnv 
from app.graders import GraderEngine 
from inference import heuristic_action 
from app.models import Action, ActionType 
env = HospitalOpsEnv('scenarios') 
scenarios = ['report_easy','report_medium','report_hard','billing_easy','billing_medium','billing_hard','bloodbank_easy','bloodbank_medium','bloodbank_hard','icu_easy','icu_medium','icu_hard'] 
for sid in scenarios: 
    obs = env.reset(sid) 
    steps = 0 
    info = None 
    while not obs.done and steps < 20: 
        od = obs.model_dump(mode='json') 
        ad = heuristic_action(od) 
        action = Action(action_type=ActionType(ad['action_type']),payload=ad.get('payload',{}),episode_id=obs.episode_id) 
        obs,reward,done,info = env.step(action) 
    steps += 1 
    score = info.grader_score if info else None 
    print(sid,'score:',score,'steps:',steps,'done:',obs.done) 
