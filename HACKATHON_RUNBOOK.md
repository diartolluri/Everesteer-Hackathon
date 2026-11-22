# Everesteer hackathon runbook

## Set up a fresh clone

The prepared VS Code workspace already has these dependencies. For a fresh clone, use Python 3.12 and run this from the repository root. LightGBM on macOS requires OpenMP. Install it with `brew install libomp` if it is missing.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

## Adapt to the challenge

Open this folder in VS Code and paste the new challenge brief into your coding agent. Use `hackathon_setup.py` to inspect the event and get its training data. Change the candidate models, features, or evaluation in `hackathon_models.py` to match the brief. Keep the saved predictor interface, which accepts the served features and returns one prediction per original row ID, to reuse `run_round.py`. Training accepts input paths, holdout and embargo sizes, and an artifact directory as command-line options.

## Open and connect

The VS Code workspace uses `.venv/bin/python` (Python 3.12). The official notebooks are under `notebooks/`. Choose the workspace `.venv` kernel to run them.

In `.env`, replace the commented `EIQ_API_KEY` line with `EIQ_API_KEY=...` using the key from Everesteer onboarding. Keep the event URL from your setup command. `.env` is ignored by Git.

Run **Everesteer | Event status** under **Run and Debug**, or use these commands from the project folder:

```sh
source .venv/bin/activate
python hackathon_setup.py status
```

The status command reads the event clock, upload allowance, target, feature encoding, scoring settings and staking availability. It also checks the platform’s published library pins for the local Python version when available. If it reports a mismatch, install the requirements URL for your Python version from the status output before packaging, and verify the local imports again.

## Download, train and compare

```sh
python hackathon_setup.py download
python hackathon_models.py
```

The download saves train, schema, scoring settings and the available training benchmark under `data/`. Training uses the last 100 expeds for the holdout and excludes the 40 expeds immediately before it from fitting. These are local defaults, and the manifest records the exact split. Training compares LightGBM, Ridge and a per-exped rank blend, then refits the candidates and saves checked cloudpickle artifacts under `artifacts/`. The manifest points to the comparison table and per-exped scores. The offline score is a proxy based on benchmark-adjusted FIT and UNQ. The platform scores INOV on the board.

To verify the pipeline without an account, run **Everesteer | Offline smoke test** or:

```sh
python hackathon_models.py --smoke-test --artifacts artifacts/smoke-test
```

## Rehearse and submit

```sh
python run_round.py practice
python run_round.py live
```

These commands are dry runs. Practice fetches `validation`. Live fetches the current round’s `live` split when intake is open. Each run checks row IDs and prediction values, and saves Parquet predictions and a JSON report under `artifacts/runs/`. Add `--model lightgbm`, `--model ridge` or `--model rank-blend` to choose individual candidates. Repeat `--model` to choose several.

Run the corresponding `--submit` command only when you are ready to send it to Everesteer. The runner checks the window again before each live upload and records the platform receipt and available scoring status.

```sh
python run_round.py practice --model lightgbm --submit
python run_round.py live --model lightgbm --submit
```

Check `python hackathon_setup.py status` for a platform-reported staking draft window, balance, slot cap and minimum. Staking decisions are manual on Everesteer.
