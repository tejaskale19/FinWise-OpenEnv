---
title: FinWise OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# FinWise OpenEnv

FinWise OpenEnv is a real-world benchmark for training and evaluating LLM agents on Indian retail portfolio advisory decisions.

Built for the Meta x PyTorch x Hugging Face OpenEnv Hackathon.

## Real-World Motivation

Indian retail investors increasingly manage equity and mutual fund portfolios directly, but common behavior patterns create avoidable risk:

- concentrated sector exposure
- under-funded SIP plans for long-term goals
- panic decisions during market drawdowns

FinWise OpenEnv models these high-impact advisory scenarios in a controlled, reproducible environment so LLM agents can be evaluated on financial reasoning, risk management, and action quality.

## Why This Benchmark Matters For LLM Evaluation

This benchmark is useful for agent evaluation because it combines:

- structured, typed state and action spaces
- multi-step planning under constraints
- deterministic grading with partial progress scoring
- realistic trade-offs between growth, risk, and liquidity

It tests whether an agent can do more than single-turn Q and A: the agent must make coherent sequential decisions with measurable financial outcomes.

## Public Hugging Face Space

Replace with your deployed public Space URL:

https://huggingface.co/spaces/ktejas19/Finwise

## OpenEnv Contract

Core environment entrypoint:

- finwise_env.env:FinWiseEnv

Required OpenEnv methods are implemented:

- reset()
- step(action)
- state()

HTTP API endpoints exposed for deployment:

- POST /reset
- POST /step
- POST /state
- GET /health

## Typed Schemas

Schemas are implemented using Pydantic in models.py.

Observation model: PortfolioObservation

- portfolio value and liquidity: cash_inr, total_portfolio_value_inr
- holdings: stocks, mutual_fund_value_inr, sip_monthly_inr
- allocation and risk: sector_exposure, risk_profile, risk_score
- goals: target_corpus_inr, projected_corpus_inr, goal_progress
- market context: nifty_trend, max_drawdown
- episode metadata: step_number, task_name, available_actions, last_action_result

Action model: PortfolioAction

- action_type: buy_stock, sell_stock, increase_sip, decrease_sip, buy_mutual_fund, sell_mutual_fund, rebalance_sector, hold
- optional fields: asset, amount_inr, target_weight, reasoning

Step output model: StepResult

- observation
- reward
- reward_breakdown
- done
- info

## Task Suite (Easy -> Medium -> Hard)

1. diversify_sector_easy
- Scenario: IT-heavy overconcentrated portfolio.
- Objective: reduce IT concentration and improve diversification.
- Max steps: 8
- Success threshold: 0.80

2. retirement_goal_medium
- Scenario: under-funded SIP for long-term retirement target.
- Objective: improve projected corpus trajectory with better SIP and allocation decisions.
- Max steps: 10
- Success threshold: 0.75

3. crash_protection_hard
- Scenario: banking-heavy drawdown with near-term liquidity pressure.
- Objective: balance capital preservation, liquidity build-up, and long-term growth.
- Max steps: 12
- Success threshold: 0.70

## Deterministic Graders And Rewarding

Each task has a deterministic grader in graders.py that:

- returns score in [0,1]
- provides partial progress scoring
- is reproducible for the same state input

Step reward is dense and composite:

- diversification improvement
- corpus growth progress
- risk reduction
- liquidity safety
- penalties for overconcentration, churn, and low cash buffer

## Environment Variables

Inference and deployment expect the following variables:

- API_BASE_URL with default: https://router.huggingface.co/v1
- MODEL_NAME with default: Qwen/Qwen2.5-72B-Instruct
- HF_TOKEN without default
- LOCAL_IMAGE_NAME optional

Example (Windows cmd):

set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
set HF_TOKEN=YOUR_TOKEN_HERE
set LOCAL_IMAGE_NAME=

## Running Inference

Run the mandatory baseline script:

python inference.py

Expected structured logs:

- [START] task=... env=... model=...
- [STEP] step=... action=... reward=... done=true|false error=...
- [END] success=true|false steps=... rewards=...

## Expected Baseline Scores

Reference baseline model: Qwen/Qwen2.5-72B-Instruct via HF router.

| Task | Difficulty | Baseline Score | Steps Used |
|------|-----------|----------------|-----------|
| diversify_sector_easy | Easy | 0.82 | 6 |
| retirement_goal_medium | Medium | 0.71 | 9 |
| crash_protection_hard | Hard | 0.58 | 12 |
| Average | - | 0.70 | - |

Scores can vary slightly by model revision and inference backend, but task behavior and grading are deterministic for fixed trajectories.

## Local Docker Run

Build image:

docker build -t finwise-openenv .

Run container:

docker run -p 7860:7860 -e HF_TOKEN=YOUR_TOKEN_HERE -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct finwise-openenv

## Hugging Face Docker Space Deployment

1. Create a new Space on Hugging Face.
2. Select SDK: Docker.
3. Push this repository to the Space.
4. Configure Space Secrets and Variables:
- HF_TOKEN (Secret)
- API_BASE_URL (Variable, optional if default used)
- MODEL_NAME (Variable, optional if default used)
- LOCAL_IMAGE_NAME (Variable, optional)
5. Confirm container starts on port 7860.
6. Verify health and environment endpoints:
- GET /health
- POST /reset
- POST /step
- POST /state

## Validation Commands

Run before submission:

- openenv validate
- python inference.py
- docker build -t finwise-openenv .

## Project Structure

- env.py
- models.py
- tasks.py
- graders.py
- finwise_env/
- app.py
- inference.py
- openenv.yaml
- pyproject.toml
- Dockerfile
- requirements.txt

## Submission Links

GitHub repository URL placeholder:

https://github.com/tejaskale19/FinWise-OpenEnv

Hugging Face Space URL placeholder:

https://huggingface.co/spaces/ktejas19/Finwise

## License

MIT
