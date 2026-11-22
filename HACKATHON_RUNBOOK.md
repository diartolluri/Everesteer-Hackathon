# Everesteer hackathon runbook

## Before the event

The prepared VS Code workspace uses `.venv/bin/python` (Python 3.12). On a fresh clone, run the following from the repository root. Install `libomp` with Homebrew on macOS if LightGBM cannot load it.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Everesteer issues an event-specific API key through onboarding or your event account page. In the setup guide, choose **Bring your own → Python SDK** for this project. Put the event key in the local `.env` as `EIQ_API_KEY` and use `EIQ_BASE_URL=https://hackathon.everesteer.ai`. The **Codex** option configures a separate MCP connection and requires the standalone `codex` command on your terminal's PATH. The Python scripts use the key from `.env`. That file, downloads, model artifacts, prediction files and local receipts are Git-ignored.

The public weekday tournament tutorial documents a separate automatic submission workflow. For the event, trust the authenticated account, schema and scoring responses, then the event-specific docs and [repository contract](AGENTS.md). [Official hackathon quickstart](https://docs.everesteer.ai/getting-started/quickstart-hackathon.md) · [Event submissions](https://docs.everesteer.ai/submissions/event-submissions.md)

```sh
source .venv/bin/activate
python -m unittest discover -s tests -v
python hackathon_models.py --smoke-test --artifacts artifacts/smoke-test
python -m scripts.event_status status --offline --artifacts artifacts/smoke-test
```

The smoke test uses synthetic data and is safe without credentials. Use the `.venv` kernel for the official notebooks under `notebooks/`. The end-to-end reference scripts and the submission section of notebook 02 can upload predictions.

GitHub Actions runs the mocked test suite offline. Everesteer’s internal obfuscation check remains scoped to its upstream repository.

## Build phase

```sh
python hackathon_setup.py status
python hackathon_setup.py download
python hackathon_models.py --baseline-only --artifacts artifacts/baseline
python -m scripts.run_round practice --artifacts artifacts/baseline
```

The first command confirms hackathon scope, the graded target, the feature encoding, scoring settings and runtime pins when published. Check the Python version and package matches before packaging. If the sandbox publishes different pins, use the URL for your Python version, reinstall those requirements, and rerun the tests and training. The local test environment and the event sandbox can use different package versions.

Once the reference candidates pass the practice dry run, use the fuller search if the clock allows:

```sh
python hackathon_models.py
python -m scripts.event_status status
python -m scripts.run_round practice
```

Defaults are three expanding walk-forward folds of 30 expeds and a separate final 100-exped chronological holdout. Each evaluation has a 40-exped embargo. Keep that split fixed when comparing model changes. A separately named sensitivity run can use `--embargo-expeds 20` if the known target horizon permits it. Keep baseline and research artifacts in separate directories.

The manifest gives the primary model, fallback order, library versions, artifact hashes, comparison table, and paths to fold and per-exped metrics. Choose by the walk-forward summaries and use the final holdout as an audit. The offline proxy covers benchmark-relative FIT and available UNQ. The server score also depends on INOV and benchmark INOV. Set the `--model` flag to a chosen alias from the manifest.

The unauthenticated scoring response can show ordinary tournament weights even on the event host. Use the authenticated scoring settings from `hackathon_setup.py`. If a saved response gives incompatible weights, the training command reports the mismatch and selects on FIT with benchmark diagnostics.

Optional research flags include `--feature-set` for a set advertised by the schema and `--max-fit-expeds` to cap fit time. An `--auxiliary-target` experiment requires a verified, embargo-safe label horizon. Use a separate `--artifacts` directory for experimental configurations. Drop the optional tuned search when a round is near and work from the proven baseline.

## Compare research candidates

Run the research comparison with the interpreter and library pins published for the event.

```sh
python -m research.research_experiments --refit-expeds 1400 --sensitivity-embargo-expeds 80
python -c 'import json; print(json.load(open("artifacts/research/manifest.json"))["primary_model"])'
python -m scripts.run_round practice --artifacts artifacts/research --candidate NAME
```

The comparison scouts configurable missing-value policies, stable and benchmark-residual feature subsets, residual targets, an optional auxiliary target, and PCA Ridge. It scales the leading configurations, sweeps rank blends and benchmark neutralization, and audits the chosen candidates on a separate 100-exped holdout. The default selection uses four chronological folds with a 40-exped embargo. The optional 80-exped check uses the same evaluation dates.

The CSV and JSON leaderboards report SDK FIT, UNQ, INOV, event score, fold variation and benchmark correlation. The full event score requires the verified core feature list and the official training benchmark. The event schema currently publishes only the core overlap count, so selection uses the labeled FIT and UNQ proxy. The oracle sweep uses historical benchmark predictions for diagnosis. Deployable candidates use feature-only inference, including a fold-trained surrogate for neutralized candidates.

The manifest records the winning name, distinct finalists and validated artifact checksums. The finalist shortlist skips repeated sets of constituents and highly correlated forecasts. Use `--candidate NAME` with the research artifact directory on either lane. Repeat the flag to enter several finalists from one split download. Live uploads require the open window and `--submit`.

For a wider search, the Ridge, residual LightGBM and categorical sweeps each produce their own model manifest and fold reports. Run them before the practice blend sweep so it can use their verified artifacts.

```sh
python -m research.research_ridge_sweep
python -m research.research_tree_sweep
python -m research.research_categorical_sweep
python -m research.research_blend_sweep
```

The targeted follow-up sweeps are `research/research_ranker_sweep.py` for within-exped ranking objectives and `research/research_residual_ridge_refinement.py` for residual Ridge variants. Both keep a later evaluation window outside model selection.

The practice blend sweep publishes batches under `artifacts/wide-blends/`. A blend receives its own server evaluation when submitted with `python -m scripts.run_round practice`. For locally validated blends, `research/research_crossfit_blends.py` fits weights on earlier historical windows and reports results on later embargoed windows and a final audit.

## Every scored round

```sh
python -m scripts.event_status status
ROUND=round_1  # Replace with the open window reported by Everesteer.
python -m scripts.run_round live --window "$ROUND" --model lightgbm
```

The live dry run downloads fresh rows and writes local Parquet predictions and a JSON report. Inspect the window, ID coverage, model checksum and upload allowance. Repeat `--model` to include fallbacks, or use the `--artifacts artifacts/baseline` flag if that is the chosen fit.

After your explicit approval for the exact lane and models, use the relevant command. For live, confirm the exact window as well:

```sh
python -m scripts.run_round practice --model lightgbm --submit
python -m scripts.run_round live --window "$ROUND" --model lightgbm --submit
python -m scripts.event_status status --window "$ROUND"
```

Check the receipt and processing status after approval and submission. The status command can also read a completed board by window. `artifacts/runs/` contains dataset and prediction checksums, model IDs, artifacts and receipts. Keep the shared local `artifacts/submissions.sqlite3` ledger when changing artifact directories. A repeat with a known receipt reuses it. For an attempt with an uncertain response, review the saved attempt and platform upload history before any retry.

## Staking decision

```sh
python -m scripts.event_status stakes --window "$ROUND"
```

This read-only command uses published availability, draft window, balance and limits. It presents an unallocated option and may show small illustrative single-model or diversified allocations when current receipts and real-data fold evidence support them. Completed-round evidence can lower confidence in a candidate. The default scenarios cap illustrative exposure at 10% of the reported balance. The live event response determines whether staking is active and what metric decides the event.

**Approval required** for any practice upload, any live upload, or any stake allocation or withdrawal. For staking, approval must specify the exact model, amount and window. The CLI provides staking analysis only.
