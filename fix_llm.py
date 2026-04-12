c = open('inference.py').read() 
c = c.replace('            except Exception:\n                action_dict = heuristic_action(obs_dict)', '            except Exception as e:\n                print(f"[LLM ERROR] {e}", flush=True)\n                raise') 
open('inference.py','w').write(c) 
print('Done') 
