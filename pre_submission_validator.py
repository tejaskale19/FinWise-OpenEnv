#!/usr/bin/env python3
"""
Run this locally before every submission.
If any assertion fails, DO NOT submit.
"""
import os
import sys
import glob

print("=== FinWise Pre-Submission Validator ===\n")
errors = []

# Check file structure
required_files = [
    "inference.py",
    "Dockerfile",
    "openenv.yaml",
    "README.md",
    "requirements.txt",
]
for f in required_files:
    if os.path.exists(f):
        print(f"✓ {f} exists")
    else:
        errors.append(f"✗ MISSING: {f}")

# Check inference.py has required elements
with open("inference.py", encoding="utf-8") as f:
    content = f.read()
    for required in ["[START]", "[STEP]", "[END]", "safe_score",
                     "API_BASE_URL", "MODEL_NAME", "HF_TOKEN",
                     "validate_all_scores"]:
        if required in content:
            print(f"✓ inference.py contains '{required}'")
        else:
            errors.append(f"✗ inference.py MISSING: '{required}'")

# Check no raw boundary returns in grader files
grader_files = glob.glob("**/*grader*.py", recursive=True) + \
               glob.glob("**/*scorer*.py", recursive=True) + \
               glob.glob("**/*eval*.py", recursive=True)

graded_seen = set()
grader_files = [g for g in grader_files if not (g in graded_seen or graded_seen.add(g))]

bad_patterns = ["return 0.0", "return 1.0", "return 0\n", "return 1\n",
                "np.clip(", "float(True)", "float(False)"]
for gf in grader_files:
    with open(gf, encoding="utf-8") as f:
        gcontent = f.read()
    for pat in bad_patterns:
        if pat in gcontent:
            errors.append(f"✗ {gf} contains dangerous pattern: '{pat}'")

# Final report
print("\n" + "="*40)
if errors:
    print("FAILED — Fix these before submitting:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Safe to submit!")
