import graders
SCORE_EPSILON = 0.005

def safe(x):
    try:
        v = float(x)
    except Exception:
        v = 0.5
    return max(0.005, min(0.995, v))

r = graders.compute_step_reward({}, {}, 'hold', 'diversify_sector_easy')[0]
print(f"raw={r} safe={safe(r)} formatted={safe(r):.2f} valid={0.0 < float(f'{safe(r):.2f}') < 1.0}")
