# Manual Annotation Tool

Local-only browser tool for Phase 1 binary annotation on a fixed 50-item shared subset.

## Current Roles

- `annotator`: 6 fixed users, each assigned 5 categories in a fixed order
- `test`: 1 fixed user, assigned 5 random categories once at initialization
- `admin`: dashboard access, raw JSONL download, and test-user reset

## Supported Categories

The active VocalGrad benchmark uses the nine categories reported in the paper.

- `volume`
- `speaking_speed`
- `voice_pitch`
- `background_noise`
- `audio_distortion`
- `audio_roughness`
- `voice_clarity`
- `voice_vibration`
- `echo`

Each category uses:

- `annotation_tool/data/manifests/<category>/shared_annotation_50.csv`

The app always loads all categories. It no longer uses
`VOCALGRAD_ANNOTATION_CATEGORY`.

## Shared Manifest Design

The shared manifests are generated from processed test manifests:

```bash
uv run python scripts/build_annotation_manifests.py
```

Current selection rule:

- 50 total items per category
- 50 distinct source clips
- one source clip per selected speaker
- the same `(source clip, difficulty, trajectory)` assignments are reused across all categories
- `difficulty x trajectory` is balanced as evenly as possible across 50 items
- `increase / decrease` is balanced overall

Reference lists:

- `annotation_tool/data/manifests/shared_source_clips_50.csv`
- `annotation_tool/data/manifests/shared_source_clips_50.txt`

## Paper Human Evaluation Protocol

The paper reports human performance collected with this web tool.

- six annotators were recruited
- each annotator was assigned up to five categories
- each category has 50 binary questions
- each item is answered by at least three annotators
- annotators may replay clips multiple times before submitting an answer
- the tool logs binary labels, replay count, response time, and timestamps

Raw per-user JSONL files and local credentials are not intended for public
release unless they are explicitly anonymized and approved. Shared manifests and
tool source code are safe to include in the reproducibility repository.

## Initialize Users

Initialize local users and passwords:

```bash
uv run python scripts/init_annotation_users.py --overwrite
```

This writes:

- hashed users file: `annotation_tool/data/users/users.json`
- plaintext credentials file: `annotation_tool/data/users/credentials.csv`

The credentials file is gitignored so you can keep usable local passwords without
committing them.

## Run The App

From the repository root:

```bash
uv run uvicorn annotation_tool.app.main:app --reload
```

From the `annotation_tool/` directory:

```bash
uv run uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Annotator Behavior

- login with `user_id` and `password`
- assigned categories are fixed
- category order is fixed
- each category has 50 items
- resume works automatically from saved JSONL rows
- only the natural-language prompt is shown during annotation
- manual playback only
- unlimited replay
- replay count and response time are logged
- sessions use signed tokens instead of server-side in-memory session state

## Admin Behavior

Admin can:

- inspect overall category-level summaries
- switch category summaries between overall / tier / trajectory views
- inspect per-annotator summaries
- inspect agreement summaries by category
- download raw JSONL files
- download all raw JSONL files as a single zip archive
- reset any non-admin user annotation data

## Results

Results are stored per category and per manifest:

```text
annotation_tool/data/results/{category}/{manifest_stem}/{annotator_id}.jsonl
```

Result writes are serialized with file locks to reduce corruption risk under
concurrent writes.

Agreement summaries are stored at:

```text
annotation_tool/data/results/{category}/{manifest_stem}/agreement_summary.json
```

## API

- `GET /`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/session/state`
- `POST /api/annotation/submit`
- `GET /api/annotator/summary`
- `GET /api/admin/dashboard`
- `GET /api/admin/download/raw`
- `GET /api/admin/download/raw-all`
- `POST /api/admin/reset-test-user`
- `POST /api/admin/reset-user`
- `GET /audio/{path:path}`
- `GET /healthz`
