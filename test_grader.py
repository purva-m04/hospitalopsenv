from app.env import HospitalOpsEnv 
from app.graders import GraderEngine 
env = HospitalOpsEnv(scenarios_dir='scenarios') 
grader = GraderEngine() 
scenarios = ['report_easy','report_medium','report_hard','billing_easy','billing_medium','billing_hard','bloodbank_easy','bloodbank_medium','bloodbank_hard','icu_easy','icu_medium','icu_hard'] 
for sid in scenarios: 
    obs = env.reset(sid) 
    score = grader.grade(env.state(), env._scenario) 
    print(sid, score, flag) 
