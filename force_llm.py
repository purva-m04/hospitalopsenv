c = open('inference.py').read() 
c = c.replace('except Exception as e:', 'except Exception as e_outer:') 
c = c.replace('action_dict = heuristic_action(obs_dict)', 'action_dict = heuristic_action(obs_dict)  # forced') 
open('inference.py','w').write(c) 
