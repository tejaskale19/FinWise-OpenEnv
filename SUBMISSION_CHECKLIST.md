# FinWise OpenEnv - Final Submission Checklist

Use this file as a final pre-submit gate for Meta OpenEnv Round 1.

## 1) Local Validation Commands

Run from repository root:

pip install -r requirements.txt
openenv validate
python inference.py

Optional local API smoke test (server running on port 7860):

python test_space.py --base-url http://localhost:7860 --task diversify_sector_easy

## 2) Docker Commands

Build:

docker build -t finwise-openenv .

Run:

docker run -p 7860:7860 -e HF_TOKEN=YOUR_TOKEN_HERE -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct finwise-openenv

Check endpoints while container is running:

curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"

## 3) Hugging Face Space Variables

Recommended configuration in Docker Space settings:

Secrets:
- HF_TOKEN

Variables:
- API_BASE_URL=https://router.huggingface.co/v1
- MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
- LOCAL_IMAGE_NAME=

Notes:
- HF_TOKEN must not have a default in inference.py.
- API_BASE_URL and MODEL_NAME may use defaults.
- LOCAL_IMAGE_NAME is optional.

## 4) Final URLs To Submit

Fill these before clicking Submit:

- GitHub Repository URL: https://github.com/<your-username>/finwise-openenv
- Hugging Face Space URL: https://huggingface.co/spaces/<your-username>/finwise-openenv

## 5) Common Failure Points

- pyproject.toml missing or invalid
- openenv validate not run before submit
- inference.py logs not in strict START/STEP/END format
- HF_TOKEN accidentally hardcoded or defaulted
- Docker daemon not running during local build test
- Space set to wrong SDK (must be Docker)
- Space secrets not configured
- /reset, /step, /state returning non-200 due to startup or env issues

## 6) Final 5 Checklist Confirmation

Mark each item only after you verify directly.

- [ ] I followed sample inference.py structure and use OpenAI client.
- [ ] inference.py contains API_BASE_URL, MODEL_NAME, HF_TOKEN, and LOCAL_IMAGE_NAME.
- [ ] Defaults are only on API_BASE_URL and MODEL_NAME (not HF_TOKEN).
- [ ] All LLM calls are via OpenAI client chat completions.
- [ ] Stdout logs follow exact START/STEP/END format.

## 7) Final Sign-off

- [ ] openenv validate passes
- [ ] python inference.py runs and logs correctly
- [ ] docker build succeeds locally
- [ ] Hugging Face Space is live and endpoint checks pass
- [ ] GitHub URL and Space URL are final and public
