import graders 
import math 
tests = [graders.grade_diversify_sector({}), graders.grade_retirement_goal({}), graders.grade_crash_protection({}), graders.grade_task('diversify_sector_easy', {})] 
all_pass = True 
for i, (score, _) in enumerate(tests): 
    ok = 0.0 < float(score) < 1.0 
    print(f'Test {i+1}: score={score} ok={ok}') 
    all_pass = all_pass and ok 
print('RESULT: READY TO SUBMIT' if all_pass else 'RESULT: STILL BROKEN') 
