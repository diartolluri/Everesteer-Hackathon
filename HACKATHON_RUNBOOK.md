# Everesteer hackathon runbook

## Before the event

The prepared VS Code workspace uses `.venv/bin/python` (Python 3.12). On a fresh clone, run the following from the repository root. Install `libomp` with Homebrew on macOS if LightGBM cannot load it.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Everesteer issues an event-specific API key as part of onboarding. In **Install & connect your agent**, open **Copy setup command** and put the event key in the local `.env` as `EIQ_API_KEY`. Use the official hackathon URL from that command. `.env`, downloads, model artifacts, prediction files and local receipts are Git-ignored.

The public weekday tournament tutorial documents a separate automatic submission workflow. For the event, trust the authenticated account, schema and scoring responses, then the event-specific docs and [repository contract](AGENTS.md). [Official hackathon quickstart](https://docs.everesteer.ai/getting-started/quickstart-hackathon.md) · [Event submissions](https://docs.everesteer.ai/submissions/event-submissions.md)

```sh
source .venv/bin/activate
python -m unittest discover -s tests -v
python hackathon_models.py --smoke-test --artifacts artifacts/smoke-test
python event_status.py status --offline --artifacts artifacts/smoke-test
```

The smoke test uses synthetic data and is safe without credentials. Use the `.venv` kernel for the official notebooks under `notebooks/`. The end-to-end reference scripts and the submission section of notebook 02 can upload predictions.

GitHub Actions runs the mocked test suite offline. Everesteer’s internal obfuscation check remains scoped to its upstream repository.

## Build phase

```sh
python hackathon_setup.py status
python hackathon_setup.py download
python hackathon_models.py --baseline-only --artifacts artifacts/baseline
python run_round.py practice --artifacts artifacts/baseline
```

The first command confirms hackathon scope, the graded target, the feature encoding, scoring settings and runtime pins when published. Check the Python version and package matches before packaging. If the sandbox publishes different pins, use the URL for your Python version, reinstall those requirements, and rerun the tests and training. The local test environment and the event sandbox can use different package versions.

Once the reference candidates pass the practice dry run, use the fuller search if the clock allows:

```sh
python hackathon_models.py
python event_status.py status
python run_round.py practice
```

Defaults are three expanding walk-forward folds of 30 expeds and a separate final 100-exped chronological holdout. Each evaluation has a 40-exped embargo. Keep that split fixed when comparing model changes. A separately named sensitivity run can use `--embargo-expeds 20` if the known target horizon permits it. Keep baseline and research artifacts in separate directories.

The manifest gives the primary model, fallback order, library versions, artifact hashes, comparison table, and paths to fold and per-exped metrics. Choose by the walk-forward summaries and use the final holdout as an audit. The offline proxy covers benchmark-relative FIT and available UNQ. The server score also depends on INOV and benchmark INOV. Set the `--model` flag to a chosen alias from the manifest.

The unauthenticated scoring response can show ordinary tournament weights even on the event host. Use the authenticated scoring settings from `hackathon_setup.py`. If a saved response gives incompatible weights, the training command reports the mismatch and selects on FIT with benchmark diagnostics.

Optional research flags include `--feature-set` for a set advertised by the schema and `--max-fit-expeds` to cap fit time. An `--auxiliary-target` experiment requires a verified, embargo-safe label horizon. Use a separate `--artifacts` directory for experimental configurations. Drop the optional tuned search when a round is near and work from the proven baseline.

## Every scored round

```sh
python event_status.py status
ROUND=round_1  # Replace with the open window reported by Everesteer.
python run_round.py live --window "$ROUND" --model lightgbm
```

The live dry run downloads fresh rows and writes local Parquet predictions and a JSON report. Inspect the window, ID coverage, model checksum and upload allowance. Repeat `--model` to include fallbacks, or use the `--artifacts artifacts/baseline` flag if that is the chosen fit.

After your explicit approval for the exact lane and models, use the relevant command. For live, confirm the exact window as well:

```sh
python run_round.py practice --model lightgbm --submit
python run_round.py live --window "$ROUND" --model lightgbm --submit
python event_status.py status --window "$ROUND"
```

Check the receipt and processing status after approval and submission. The status command can also read a completed board by window. `artifacts/runs/` contains dataset and prediction checksums, model IDs, artifacts and receipts. Keep the shared local `artifacts/submissions.sqlite3` ledger when changing artifact directories. A repeat with a known receipt reuses it. For an attempt with an uncertain response, review the saved attempt and platform upload history before any retry.

## Staking decision

```sh
python event_status.py stakes --window "$ROUND"
```

This read-only command uses published availability, draft window, balance and limits. It presents an unallocated option and may show small illustrative single-model or diversified allocations when current receipts and real-data fold evidence support them. Completed-round evidence can lower confidence in a candidate. The default scenarios cap illustrative exposure at 10% of the reported balance. The live event response determines whether staking is active and what metric decides the event.

**Approval required** for any practice upload, any live upload, or any stake allocation or withdrawal. For staking, approval must specify the exact model, amount and window. The CLI provides staking analysis only.
