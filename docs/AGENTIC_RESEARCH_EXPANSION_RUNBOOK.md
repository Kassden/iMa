# Isolated V6 Research Runbook

The parent owns config/controller and coordinates cutover. The user authorized
isolated dependency setup, exact committed source transfer/read-only extraction,
Linux fixture canary and a guarded official dataset build. These operations
have been performed; the first official build was explicitly cancelled after
parent-confirmed builder defects. Only dedicated own-user
fixture/build transient units were started. V5 STOP/configuration/unit mutation,
acquisition mutation, paid planning and production V6 launch remain held pending
the repaired release and acceptance gates. Existing effective V5 dependency
versions were preserved; its three missing requirements were repaired privately.

Subsequent assignment authorized exact committed source
`5b63862360f5380b6f247bf1c4da0a0854528aa6`, separate bounded fixture and
official dataset builder only. Transfer/read-only extraction completed; Linux
fixture ledger has five completions and delivered tracking/model registrations,
but MLflow runs are FAILED with temporary code-cleanup permission errors and a
trace error. This is not a clean tracking gate. Dataset build ran under
8,000,000,000 bytes/two CPU quota and was cancelled at 08:41:16 UTC; no candidate
was published or accepted. The 1800s request budget and actual hard timeout were
1800s; attempts to apply the subsequently specified 3600s timeout did not load.
Future builds must read back the intended 3600s before starting. See validation for
proof paths; V5 STOP, V5 unit/config change, paid planner and live V6 launch
remain explicitly unauthorized. Earlier prepared-only sections are historical
recipes, not statements that the assigned transfer/fixture never occurred.
GitHub CI on that SHA is currently failed (parent identified two clean-checkout
archive-fixture assumptions). Parent owns the corrected/new committed release.
Do not cut over, reuse the cancelled staging data, or relabel its provenance.
Parent identified raw-manifest variable shadowing and event distance unit metadata
defects; await a repaired committed release and fresh request v3. Preserve old
request/staging and cancellation evidence. The optional full-row preparation
benchmark was NOT run: it requires a valid immutable candidate first.

Authorized frozen source copy is now at
`campaigns/agentic_v6_research/inputs/official-snapshot`: 17 regular files,
776,217,339 bytes, all hashes verified and content read-only. Its manifest SHA256
is unchanged; `inputs/official-snapshot-copy-receipt.json` records both paths and
all file hashes. Preserve the dataset manifest's original acquisition source path;
the parent launcher is adding hash-verified relocation support, not a path rewrite.

## Identity And Integration API

Host/account: `imaopt@100.95.24.121`. Keep pinned V5 campaign, release, dependencies
and service unchanged. Use a separate `agentic_v6_*` campaign with immutable
release, dataset, protocol and environment identities. The existing verified CLI
invocation is:

```bash
"$IMA_V6_PYTHON" -m scripts.optimize run --campaign "$IMA_V6_CAMPAIGN" --config "$IMA_V6_CONFIG"
```

Successor policy is `expansion_v6` in the parent's `CampaignConfig`/CLI. The
launcher imports `scripts.optimize._load_config(path)` and calls
`ima.optimizer.CampaignConfig(campaign_dir=campaign, **values).validate()` before
execution. Config stays schema version 1 and uses the existing CLI's alias/strict
key validation. A valid policy enum alone does not prove the successor execution
path is wired: verify `run_campaign` dispatch and a deterministic canary too.

Parent installs `deploy/systemd/ima-research-expansion-supervisor.service` as an
imaopt user unit and creates mode-0600
`/home/imaopt/.config/imaopt/research-expansion.env` with these non-secret values:

```ini
IMA_V6_RELEASE=/home/imaopt/research-v2/live-releases/ima-v6-<release-id>
IMA_V6_REVISION=<exact-release-revision>
IMA_V6_PYTHON=/home/imaopt/research-v2/v6-dependencies/venv/bin/python
IMA_V6_DEPENDENCIES=/home/imaopt/research-v2/v6-dependencies/site-packages
IMA_V6_CAMPAIGN=/home/imaopt/research-v2/campaigns/agentic_v6_research
IMA_V6_CONFIG=/home/imaopt/research-v2/campaigns/agentic_v6_research/ops/openrouter-config.json
IMA_V6_RESEARCH_POLICY=expansion_v6
```

Replace placeholders with approved immutable values; they are not runnable
defaults. Interpreter and overlay must exist in an isolated successor environment.
Release must be an existing directory, not a mutable symlink, with an exact
`REVISION` match. Release/dependency/config/campaign paths resolve inside imaopt's
`research-v2` root. The interpreter entry-point directory belongs to that root;
its executable may resolve to the verified own runtime under `/home/imaopt/python`.
config belongs to the campaign; campaign is a direct `campaigns/agentic_v6_*`
child; existing absolute dataset/protocol files are mandatory. Copy verified
official inputs into immutable own release/campaign inputs if needed, preserving
their manifest/lineage and hashes. Never point campaign inputs at `latest_verified`.

The unit executes `/bin/bash ${IMA_V6_RELEASE}/deploy/systemd/ima-research-expansion-supervisor`.
The script needs no executable bit. It sets `PYTHONPATH`, disables user-site
packages and bytecode writes, sets four native thread limits to one and exports
`IMA_CODE_REVISION`.
Run mode sources only the existing own-user
`/home/imaopt/.config/imaopt/openrouter.env` without printing credentials. Do not
use shell tracing or dump full process/config environments. Output goes to the
own user journal. HTTP timeout/retry are bounded as in V5.

## Preparation Gates

Paid production deployment is held until the parent records passing implementation,
data, resource/recovery and repaired committed-release gates. Routine choices need
no additional approval. Isolated dependency setup, assigned committed source
transfer, and dedicated fixture/build transient units are authorized and have run;
protected service STOP/reload/start/configuration changes are not authorized.
The original strict-lock recipe below is a separate option:
the current copy-mode assignment preserves effective Pydantic 2.13.5/core 2.46.5
and must record this departure from the partial research lock explicitly.

1. Record the immutable V5/protected-unit hashes from the validation document,
   live own-user unit state and MLflow health/experiment readback. Check own disk
   headroom and the active cgroup's byte limits, current/peak/events/pressure.
2. Pass the legacy plus successor suite and the deterministic expansion canary.
   Verify existing `run`, `status`, `stop` CLI help on the actual release. The
   canary CLI is parent-owned; its help passes locally. This role executed the
   five-trial Linux variant with private SQLite: ledger passed, MLflow runs failed
   as documented in validation. The ten-trial example below is a recipe, not that
   actual invocation. V5's canary is not V6 proof:

```bash
"$IMA_V6_PYTHON" -m scripts.run_research_expansion_canary --output <fresh-isolated-canary-directory> --planner fixture --max-concurrent-trials 2 --max-trials 10
```

   Require nonzero completed work, the intended trial count, bounded cost,
   preparation/feedback evidence and clean final state, not just exit zero.
   The updated script enforces complete mode, exact requested count, no failures
   or pending tells, and no pending tracking when MLflow is configured. Offline
   tracking outbox entries are expected and must not be mislabeled delivered.
   An authenticated
   `--planner openrouter --model deepseek/deepseek-v4.1-flash` run is a separate
   paid gate; this script uses a $1 canary cap, not the campaign's $5 cap.
3. Transfer source/wheels through approved Tailscale routing. Install existing
   locks and extras into the isolated V6 environment; run `pip check`, capture
   resolved versions, freeze manifest and compute environment identity. Existing
   lock files are partial; preserve V5's drift rather than updating it in place.
4. Freeze the new dataset and compatible V6 protocol. Audit the 14,488 raw races,
   14,471 feature races, attrition/exclusions, source timing and retained history.
   V5's 17,000 minimum cannot run on this snapshot. Record changed folds/population
   and distinguish original champion replay from controls scored on V6 races.
5. Verify wide-feature, chronological graph, speed/distribution, resource,
   tracking/cost and paper-EV/Kelly gates listed in validation. No missing quote
   or publication proof may be converted into invented values.
6. Run launcher `--check` with the non-secret deployment variables exported:

```bash
bash "$IMA_V6_RELEASE/deploy/systemd/ima-research-expansion-supervisor" --check
```

`--check` validates paths, revision and config without loading credentials,
creating campaign state or starting the optimizer. It does not validate data
content, temporal folds, remote planner availability, graph execution or replay.
Those gates remain separate. A retained campaign `STOP` marker prevents run mode
from launching; do not erase it automatically.

## Committed Deployment Package

Build from the full committed SHA whose exact code/config passed the parent's
gates. Do not archive the workspace with `tar`, rsync dirty source files or use an
uncommitted config. Tests on a dirty tree do not certify a committed release.
The verifier below uses standard-library tar parsing and Git object IDs; it
does not extract the archive, import campaign code, read credentials or deploy.

```bash
RESEARCH_RELEASE_SHA=<full-40-character-committed-SHA>
RESEARCH_RELEASE_ARCHIVE=<local-artifact-directory>/ima-v6-$RESEARCH_RELEASE_SHA.tar
git archive --format=tar --output "$RESEARCH_RELEASE_ARCHIVE" "$RESEARCH_RELEASE_SHA" -- ima scrapper scripts config tests deploy/systemd pyproject.toml requirements-research.lock requirements-v5-overlay.lock docs/AGENTIC_RESEARCH_EXPANSION_PLAN.md docs/AGENTIC_RESEARCH_EXPANSION_VALIDATION.md docs/AGENTIC_RESEARCH_EXPANSION_RUNBOOK.md docs/RESEARCH_NOMENCLATURE.md
bash deploy/systemd/ima-research-expansion-supervisor --verify-package "$RESEARCH_RELEASE_ARCHIVE" "$RESEARCH_RELEASE_SHA"
```

Run from a repository containing that commit. No archive prefix or compression
is accepted by this verifier. It checks the Git archive PAX commit ID, complete
scoped committed file set, blob IDs, executable modes and mandatory V6 paths.
It rejects linked/duplicate/unexpected files, path traversal, wrong revisions,
tampering and missing controller/config/service/test/doc artifacts. Success emits
archive SHA256, byte length and committed file count; execution gates remain
`not_checked`. CI performs this check after the regression suite against its own
committed `github.sha`; it performs no upload or deploy.

The parent must commit the six owned files plus its implementation and final
`config/agentic_research_expansion.json` before a complete package exists. This
role does not stage or commit. A baseline archive can pass source integrity but
still fail required V6 paths. Generate `REVISION` from the verified committed SHA
only when the parent materializes the own immutable release; treat that file,
dependency manifest and campaign inputs as explicit generated release metadata.
After permitted transfer over `imaopt@100.95.24.121`, verify the identical archive
checksum/length before extraction; preserve package receipt and source tree hashes.
Use a fresh successor directory, never unpack over V5. Refresh the protection
baseline immediately before any later cutover.

### Exact Transfer And Immutable Readback

This procedure was assigned and completed for exact SHA
`5b63862360f5380b6f247bf1c4da0a0854528aa6`; validation records its receipts.
A future repaired release requires its own exact SHA, gates and fresh receipts.
Current dirty/untracked source is never an archive input. No V5 STOP, unit action,
credential source or live planning appears in this sequence. Run local packaging
with `set -euo pipefail`; a failed readiness check must stop before any SSH/scp.
The committed supervisor must include `committed_tree` in its success receipt.

```bash
set -euo pipefail
: "${RESEARCH_RELEASE_SHA:?Set parent-supplied full committed SHA}"
: "${RESEARCH_ARTIFACT_DIR:?Set fresh local artifact directory}"
[[ $RESEARCH_RELEASE_SHA =~ ^[0-9a-f]{40}$ ]]
git rev-parse --verify "$RESEARCH_RELEASE_SHA^{commit}"
mkdir -m 700 "$RESEARCH_ARTIFACT_DIR"
git show "$RESEARCH_RELEASE_SHA:deploy/systemd/ima-research-expansion-supervisor" > "$RESEARCH_ARTIFACT_DIR/package-verifier"
RESEARCH_RELEASE_ARCHIVE="$RESEARCH_ARTIFACT_DIR/ima-v6-$RESEARCH_RELEASE_SHA.tar"
git archive --format=tar --output "$RESEARCH_RELEASE_ARCHIVE" "$RESEARCH_RELEASE_SHA" -- ima scrapper scripts config tests deploy/systemd pyproject.toml requirements-research.lock requirements-v5-overlay.lock docs/AGENTIC_RESEARCH_EXPANSION_PLAN.md docs/AGENTIC_RESEARCH_EXPANSION_VALIDATION.md docs/AGENTIC_RESEARCH_EXPANSION_RUNBOOK.md docs/RESEARCH_NOMENCLATURE.md
IMA_V6_PYTHON=python3 bash "$RESEARCH_ARTIFACT_DIR/package-verifier" --verify-package "$RESEARCH_RELEASE_ARCHIVE" "$RESEARCH_RELEASE_SHA" > "$RESEARCH_ARTIFACT_DIR/package-receipt.json"
RESEARCH_RECEIPT_SHA=$(shasum -a 256 "$RESEARCH_ARTIFACT_DIR/package-receipt.json" | cut -d ' ' -f 1)
ssh imaopt@100.95.24.121 "/home/imaopt/research-v2/v6-dependencies/venv/bin/python -I -B -c 'import re,sys; from pathlib import Path; s=sys.argv[1]; assert re.fullmatch(r\"[0-9a-f]{40}\",s); p=Path(\"/home/imaopt/research-v2/v6-staging\"); p.mkdir(mode=0o700,exist_ok=True); assert p.resolve()==p; (p/s).mkdir(mode=0o700)' '$RESEARCH_RELEASE_SHA'"
scp "$RESEARCH_RELEASE_ARCHIVE" "imaopt@100.95.24.121:/home/imaopt/research-v2/v6-staging/$RESEARCH_RELEASE_SHA/package.tar"
scp "$RESEARCH_ARTIFACT_DIR/package-receipt.json" "imaopt@100.95.24.121:/home/imaopt/research-v2/v6-staging/$RESEARCH_RELEASE_SHA/package-receipt.json"
ssh imaopt@100.95.24.121 "/home/imaopt/research-v2/v6-dependencies/venv/bin/python -I -B - '$RESEARCH_RELEASE_SHA' '$RESEARCH_RECEIPT_SHA'" <<'PY_RECEIVE'
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

revision, receipt_hash = sys.argv[1:]
assert re.fullmatch(r'[0-9a-f]{40}', revision)
assert re.fullmatch(r'[0-9a-f]{64}', receipt_hash)
base = Path('/home/imaopt/research-v2')
stage = base / 'v6-staging' / revision
release = base / 'live-releases' / ('ima-v6-' + revision)
assert stage.resolve() == stage and release.parent.resolve() == release.parent
receipt_file, archive = stage / 'package-receipt.json', stage / 'package.tar'
assert not receipt_file.is_symlink() and not archive.is_symlink()
assert hashlib.sha256(receipt_file.read_bytes()).hexdigest() == receipt_hash
receipt = json.loads(receipt_file.read_text())
assert receipt['commit'] == revision and receipt['source_integrity'] == 'passed'
assert receipt['required_paths'] == 'passed'
assert archive.stat().st_size == receipt['archive_bytes']
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['archive_sha256']
expected, seen = receipt['committed_tree'], set()
assert len(expected) == receipt['committed_files'] and expected
assert 'REVISION' not in expected and 'source-package-receipt.json' not in expected
with tarfile.open(archive, mode='r:') as package:
    assert package.pax_headers.get('comment') == revision
    for member in package:
        path, name = PurePosixPath(member.name), member.name.rstrip('/')
        assert not path.is_absolute() and '..' not in path.parts
        if member.isdir():
            assert any(p.startswith(name + '/') for p in expected)
            continue
        assert member.isfile() and name in expected and name not in seen
        mode, oid = expected[name]
        assert mode in {'100644', '100755'}
        assert bool(member.mode & 0o111) == (mode == '100755')
        blob = subprocess.check_output(['git', 'hash-object', '--stdin'],
                                       input=package.extractfile(member).read()).decode().strip()
        assert blob == oid
        seen.add(name)
    assert seen == set(expected)
    os.umask(0o077)
    release.mkdir(mode=0o700, exist_ok=False)
    package.extractall(release, filter='data')
assert {str(p.relative_to(release)) for p in release.rglob('*') if p.is_file()} == set(expected)
for name, (mode, oid) in expected.items():
    path = release / name
    assert not path.is_symlink()
    assert bool(path.stat().st_mode & 0o111) == (mode == '100755')
    assert subprocess.check_output(['git', 'hash-object', '--stdin'],
                                   input=path.read_bytes()).decode().strip() == oid
    path.chmod(0o555 if mode == '100755' else 0o444)
(release / 'REVISION').write_text(revision + '\n')
(release / 'source-package-receipt.json').write_bytes(receipt_file.read_bytes())
for name in ('REVISION', 'source-package-receipt.json'):
    (release / name).chmod(0o444)
for path in sorted(release.rglob('*'), key=lambda p: len(p.parts), reverse=True):
    if path.is_dir():
        path.chmod(0o555)
release.chmod(0o555)
print(json.dumps({'release': str(release), 'commit': revision,
                  'committed_files': len(seen), 'receipt_sha256': receipt_hash,
                  'archive_sha256': receipt['archive_sha256'],
                  'pax_tree_and_extracted_readback': 'passed', 'launch': 'not_performed'}))
PY_RECEIVE
```

Target Git `hash-object` is used outside a repository; no clone, Git index,
branch change or history transfer is required. Archive and receipt are retained
in the SHA-specific staging directory. Receiver refuses reused destination,
linked paths, receipt/content/mode/tree tampering before extraction. A failed
post-extraction check leaves a rejected candidate for review, never promotion;
do not delete/reuse it automatically. `REVISION` and the copied receipt are the
only generated release metadata; all committed files/directories become
read-only. No mutable alias is created. The parent must independently refresh
V5 protection before/after any actual transfer and provide the Linux fixture
command, bounded destination/resources and expected completion/tracking counts.
Native-thread caps, private dependency `PYTHONPATH` and bytecode-disabled imports
are required; verify `ima` and `scripts` import from this committed release.

## Guarded Server Dataset Build: Prepared, Not Started

Read-only feasibility at `2026-10-04T07:23:58.863161+00:00`: own user manager
is running, systemd 259.5, `Delegate=yes` with cpu/memory/pids; `systemd-run
--user --help` supports the required properties/wait/pipe/working-directory/env.
Host: 28 CPUs, load average 9.64/9.84/9.45, 130,391,482,368 physical bytes,
29,046,710,272 MemAvailable, 5,423,894,528 MemFree. Available includes reclaimable
cache; it is not idle anonymous RAM. V5 current 98,719,813,632, peak
103,083,323,392, max 107,374,182,400 bytes; ancestor memory/cpu ceilings are
unlimited. The build is a sibling unit, never inside V5's service cgroup.
Manager ancestors have historical OOM/kill events (16/7), and host memory
pressure full avg10/60 was 0.06/0.08. Do not erase or call those counters clean.

Conservative admission needs build cap 8,000,000,000 + host reserve
8,000,000,000 + possible V5 growth to its cap. At this sample it needs
24,654,368,768 available bytes and leaves 4,392,341,504 above that allowance:
provisionally feasible, not permission or measured build success. Recheck twice
immediately before launch and abandon if a sample fails. No transient probe
unit was created, so successful unit creation/enforcement remains untested.

Parent must first provide/test the committed SHA, an approved immutable request,
fixed source snapshot/manifest and separate V6 registry/output. This role verified
local `python -m scripts.build_research_dataset --help`; it requires `--request`,
`--registry`, `--source-snapshot`, `--raw-manifest`. The registry path must be V6,
never V5/acquisition outputs. Import inspection of the candidate builder uses
NumPy/pandas and existing feature/event modules; raw-page acquisition rebuilds
remain separate and need their audited acquisition extras.

Prepared own-user guard (read-only itself; no service action):

```bash
"$IMA_V6_PYTHON" -I -B - <<'PY_BUILD_GUARD'
import json
import subprocess
import time
from pathlib import Path

for sample in range(2):
    mem = {line.split(':', 1)[0]: int(line.split()[1]) * 1024
           for line in Path('/proc/meminfo').read_text().splitlines()
           if line.startswith('MemAvailable:')}
    cg = subprocess.check_output(['systemctl', '--user', 'show',
        'ima-discovery-v5-supervisor.service', '-p', 'ControlGroup', '--value'], text=True).strip()
    path = Path('/sys/fs/cgroup') / cg.lstrip('/')
    current, maximum = int((path / 'memory.current').read_text()), int((path / 'memory.max').read_text())
    required = 16000000000 + max(0, maximum - current)
    full = next(line for line in Path('/proc/pressure/memory').read_text().splitlines()
                if line.startswith('full '))
    avg10 = float(dict(item.split('=') for item in full.split()[1:])['avg10'])
    assert mem['MemAvailable'] >= required and avg10 < 0.5, 'Host headroom/pressure gate failed'
    parent = path.parent
    while parent != Path('/sys/fs/cgroup'):
        limit = (parent / 'memory.max').read_text().strip()
        if limit != 'max':
            assert int(limit) - int((parent / 'memory.current').read_text()) >= required, 'Ancestor headroom gate failed'
        parent = parent.parent
    print(json.dumps({'sample': sample, 'available_bytes': mem['MemAvailable'],
                      'required_bytes': required, 'v5_current_bytes': current,
                      'host_full_pressure_avg10': avg10, 'verdict': 'admissible_now'}))
    if sample == 0:
        time.sleep(5)
PY_BUILD_GUARD
```

The 0.5% avg10 pressure cutoff is an operational guard, not a measured guarantee.
Even a pass can become stale. Run the following ONLY after parent assignment,
with `set -euo pipefail` so a failed guard prevents the launch. All placeholders
are required non-secret parent choices, not defaults or existing output claims:

```bash
systemd-run --user --no-ask-password --service-type=exec --wait --pipe --unit="ima-v6-dataset-build-$RESEARCH_RELEASE_SHA" --working-directory="$IMA_V6_RELEASE" --property=MemoryAccounting=yes --property=CPUAccounting=yes --property=MemoryHigh=6000000000 --property=MemoryMax=8000000000 --property=MemorySwapMax=0 --property=CPUQuota=200% --property=TasksMax=128 --property=Nice=10 --property=RuntimeMaxSec=3600 --property=TimeoutStopSec=60 --setenv="PYTHONPATH=$IMA_V6_DEPENDENCIES:$IMA_V6_RELEASE" --setenv=PYTHONNOUSERSITE=1 --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 "$IMA_V6_PYTHON" -B -m scripts.build_research_dataset --request "$IMA_V6_DATASET_REQUEST" --registry "$IMA_V6_CAMPAIGN/datasets" --source-snapshot /home/imaopt/acquisition/hkjc-20261002/snapshots/snapshot-20261004T050721Z-122062e8 --raw-manifest /home/imaopt/acquisition/hkjc-20261002/snapshots/snapshot-20261004T050721Z-122062e8/manifest.json
```

Capture exit/stdout/stderr and invocation; read back the transient unit's actual
8,000,000,000-byte MemoryMax and 2-second-per-second CPU quota while running,
plus cgroup peak/events/pressure. An OOM, timeout, unsupported source requirement
or incomplete candidate must remain failed/unpromoted; preserve staging evidence.
No V5 STOP, signal, limit update or restart is part of building. The final dataset
must still pass historical retention, chronology/protocol/coverage gates before
the campaign can use it. Do not treat an 8 GB cap as proof the build fits it.

## Candidate Config Example

Parent now supplied `config/agentic_research_expansion.json` and
`scripts/run_research_expansion_canary.py` in the working tree. Local real-loader
and `CampaignConfig.validate()` checks passed without executing a campaign.
The approved config uses `expansion_v6`, `planner_mode=openrouter`,
`model=deepseek/deepseek-v4.1-flash`, `max_total_cost_usd=5`,
`max_concurrent_trials=26`, `cpu_thread_budget=24`,
`ram_budget_gib=93.13225746154785`, `memory_budget_gb_decimal=100`,
two active preparations and four CPU/eight GiB host reserves. Its
`max_trials_per_decision=260` alias normalizes to `proposal_batch_size=260`.
Campaign is `agentic_v6_research`; inputs are `inputs/features.parquet` and
`inputs/protocol.json`, registry `datasets`, snapshot `inputs/official-snapshot`;
reference campaign remains the pinned `agentic_v5_discovery`. All are under
`/home/imaopt/research-v2/campaigns`. Schema success does NOT certify file presence,
compatible folds, paid access, controller integration or memory admission.
Twenty-six is a ceiling, not a promise to admit 26 simultaneous workers inside
the 24-thread/resource budgets. Do not replace the approved model with a blank
or unknown identifier.

The earlier fixture-only example below is retained as a schema example, tested
with the CLI constructor mocked before execution. It is not a launch config,
resource benchmark or protocol decision; replace the illustrative paths only
after the parent chooses/fixes the relevant contracts.

```json
{
  "schema_version": 1,
  "policy": "agentic",
  "research_policy": "expansion_v6",
  "planner_mode": "fixture",
  "model": "openrouter/local-policy",
  "max_trials": null,
  "proposal_batch_size": 260,
  "max_concurrent_trials": 1,
  "max_new_programs_per_decision": 12,
  "max_pending_programs": 64,
  "max_active_preparations": 1,
  "max_inflight_programs": 12,
  "queue_low_watermark": 8,
  "cpu_thread_budget": 2,
  "ram_budget_gib": 16,
  "host_reserve_cpu_threads": 4,
  "host_reserve_ram_gib": 8,
  "memory_budget_gb_decimal": 100,
  "worker_max_tasks": 1,
  "planning_checkpoint_seconds": 300,
  "replan_every_terminal_trials": 32,
  "max_consecutive_failed_trials": 8,
  "dataset_path": "/home/imaopt/research-v2/campaigns/agentic_v6_example/inputs/features.csv",
  "protocol_path": "/home/imaopt/research-v2/campaigns/agentic_v6_example/inputs/protocol.json",
  "dataset_registry_path": "/home/imaopt/research-v2/campaigns/agentic_v6_example/datasets",
  "official_snapshot_path": "/home/imaopt/research-v2/campaigns/agentic_v6_example/inputs/official-snapshot",
  "mlflow_tracking_uri": "http://100.95.24.121:5000"
}
```

For real OpenRouter planning the parent must set `planner_mode=openrouter`, an
explicit model and positive `max_total_cost_usd`; fixture schema validation cannot
prove paid decisions or feedback. `proposal_batch_size` is a proposal ceiling,
not worker concurrency. `max_trials_per_decision` remains the CLI alias and must
not conflict with it. Existing CLI fields distinguish RAM GiB from the decimal
process-group `memory_budget_gb_decimal`. The example's one-worker/16-GiB values
are illustrative and do not prove headroom alongside active V5.

## Own-User Dependency And Route Inventory

Readback `2026-10-04T06:11:11.309911Z`: own-user research base venvs under
`/home/imaopt/research-v2/releases/<id>/.venv/bin/python`, Python 3.12.12:

| Base family | Available IDs | Difference |
|---|---|---|
| Modeling/research stack with CatBoost/LightGBM | `0b72703`, `10be179`, `2868bec`, `4c668e0`, `87de1c5`, `883fa15`, `90d20fb`, `93bf45a`, `a6932b5`, `d9e35c5`, `f54c604` | No features/httpx/acquisition/browser extras |
| Research stack without CatBoost/LightGBM | `16614cc`, `4322124`, `55ac4b8`, `8c2f9d806469953b2033d7c906f4e3392a436022`, `c70e970da6638316434ad68d8c7735d035f2c90e`, `f383ab0b` | Also missing the two model backends |
| Own live-release base venvs | `084fdb3`, `15b4702`, `272a192`, `2f86edb`, `defeede` | Same modeling-stack versions/missing extras as `f54c604` |

Research stack versions match the validation baseline. In particular pyarrow
25.0.1, threadpoolctl 3.7.0 and statsmodels 0.15.0 are available. Base venvs lack
Featuretools, feature-engine, Woodwork, httpx, Scrapy, pypdf and Playwright; V5's
overlay supplies the first four but not acquisition/browser extras. NumExpr and
NGBoost are absent and optional; do not add them without a measured requirement.

Own acquisition venvs under `/home/imaopt/acquisition/hkjc-20261002`:
`snapshot-venv` contains the data stack plus Scrapy 2.19.0;
`snapshot-venv-r5` additionally has pypdf 6.19.0. These lack the research/model/
feature/httpx/browser extras. `venv` has Scrapy; `venv-r16` has Scrapy plus pypdf
and lacks the data/model stack. They serve acquisition and must not be modified.

The base `f54c604` and `research-v2` root contain the research lock but no V5
overlay lock. Both locks exist in the pinned `ima-v5-current` source. The research
lock asks for Pydantic 2.13.4 while those venvs report 2.13.5; retain that difference
in the V6 dependency receipt. Own wheel stores: `research-v2/wheelhouse`, 12 wheels
/ 23,423,423 bytes, and `v5-dependencies/wheels`, 26 wheels / 92,909,496 bytes.
They contain some research/feature/httpx wheels but are not a proven complete V6
wheelhouse. Prepare a separate successor env; do not pip-install into a protected
base, overlay or acquisition venv. Copying a venv without verifying absolute
paths/entry points is not a validated successor environment.

The `f54c604/.venv/bin` directory actually resolves through the own
`releases/2868bec/.venv/bin` base, and its Python executable resolves to
`/home/imaopt/python/cpython-3.12.12/bin/python3.12`. Preserve this shared base;
the parent may reuse its read-only interpreter with a separate verified V6
overlay, or provision a new successor venv. Record effective site-packages,
interpreter/entry-point paths and resolved versions in either case. Do not assume
separate release-directory names imply separate Python environments.

Local route to `100.95.24.121` is `utun11`, MTU 1280. On the remote own account,
`openrouter.ai` resolved to `104.18.2.115` and `104.18.3.115`, both routed via
`tailscale0`, table 52, source `100.95.24.121`, uid 1001. Endpoint is
`https://openrouter.ai/api/v1/chat/completions`. The unauthenticated public
`/api/v1/models` request returned 200. This verifies public egress, not model
credits, authenticated requests or provider latency. DNS/routes can change;
refresh them immediately before later launch. No route or proxy config was changed.

`openrouter.env` is own-user mode 0600, owner uid 1001. A read-only shell sourced
it and reported only `OPENROUTER_API_KEY` nonempty; no value or length was printed.
Use this exact own-user source path and no shell tracing. `research-expansion.env`
was absent. The unit never embeds a key or passes it on a command line.

## Isolated Offline Dependency Setup

User subsequently authorized copy-mode setup, only under
`/home/imaopt/research-v2/v6-dependencies`. Completed at
`2026-10-04T06:49:26.590593+00:00`: fresh venv and copied base/overlay,
all 120 original effective versions preserved, 189 effective distributions,
25 private import origins, target offline dry-run/install/pip check exit 0.
All 31,187 copied files passed per-file content checks before repair.
Locally downloaded 95 wheels / 59,588,314 bytes were transferred over Tailscale
and every hash verified before no-index installation. Only the three missing
requirements and their absent closure were installed; no server downloads or
service changes. All ten protected hashes, V5 PID/invocation and STOP absence
matched afterward. Actual command forms (already executed with concrete paths):

```bash
.venv/bin/python -m pip download --only-binary=:all: --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 --implementation cp --python-version 3.12 --abi cp312 --dest "$V6_WHEELS" --constraint "$V5_EFFECTIVE_CONSTRAINTS" 'jupyterlab==4.6.2' 'graphviz==0.21' 'plotly==7.1.0'
scp -r "$V6_WHEELS" imaopt@100.95.24.121:/home/imaopt/research-v2/v6-dependencies/
PYTHONPATH=/home/imaopt/research-v2/v6-dependencies/site-packages PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_CONFIG_FILE=/dev/null /home/imaopt/research-v2/v6-dependencies/venv/bin/python -m pip install --no-cache-dir --no-index --find-links /home/imaopt/research-v2/v6-dependencies/wheels --constraint /home/imaopt/research-v2/v6-dependencies/preserve.constraints --report /home/imaopt/research-v2/v6-dependencies/evidence/offline-install.json 'jupyterlab==4.6.2' 'graphviz==0.21' 'plotly==7.1.0'
```

Local artifact root was
`/var/folders/wq/tn0d0r417cb3kghxltyqhjt80000gn/T/ima-v6-dependency-nfba8fuf`;
baseline manifest/constraint generation used read-only SSH metadata output.
Private evidence is under `v6-dependencies/evidence`: resolved effective JSON/
lock, import origins, complete file/link inventory, copy integrity, wheel
manifest, pip plan/install reports and logs, pip check/freeze, pre/post protection
receipts and indexed checksums. Exact hashes and byte counts are in validation.
Environment-files SHA256 is
`2dd0482463a56e33c84dc11fe2174469a5cc2451efaa7c0ecd4bad0d9311ab4c`;
setup receipt SHA256
`3a108b4002b5c8f17dce43912f15e09ac41dae2bd6e78b8bda76185c3fb90ed6`.
This mode deliberately preserves Pydantic 2.13.5/core 2.46.5 rather than claiming
strict partial-lock parity. Strict-lock/acquisition instructions below remain
future alternatives and were not executed in this copy-mode assignment.

Parent may now use the private interpreter and overlay env values from the
integration API. Source must still come from a gated committed `git archive`;
verify `ima`/`scripts` resolve to that release, not copied project metadata.
No full official acquisition/browser extras or advanced-runtime proof is implied.
Paid production launch, V5 STOP and production unit installation remain held;
the separately authorized fixture/build transient units have run.

Read-only receipt refresh after compaction confirmed the existing setup/pip-check
hashes and a fresh private `pip check` exit 0; no new remote writes. Parent's
`.tmp/v6-canary-tracking-1` local fixture now has five FINISHED MLflow runs,
five READY registered versions linked to those runs, four OK traces and zero
pending tracking/tells. Status/SQLite readback is recorded in validation.
This is local fixture tracking proof, not remote/paid/native-USD-UI acceptance.
Before a Linux code canary, require the parent source freeze/committed archive,
unchanged private environment identity, verified import resolution to that
release and a separately bounded fixture destination. Official-data and live
controller gates remain prerequisites to launch, not conditions waived by the
private dependency or local tracking pass.

Read-only refresh `2026-10-04T06:26:24.957593Z`: effective V5 metadata
manifest (120 distributions, first import-path match) SHA256
`97dd8c60d6edda3180a9ae0a586c6d98a1f6c837f8f0ae2f5570da9608bf5bd9`.
Resolved base purelib is
`/home/imaopt/research-v2/releases/2868bec/.venv/lib/python3.12/site-packages`:
27,197 files / 1,150,969,295 bytes, no symlinks or `.pth`/egg-link/customization
hooks. V5 overlay: 3,990 files / 28,858,761 bytes, no symlinks; its only `.pth`
is setuptools' 151-byte `distutils-precedence.pth`. Record fresh hashes before
copying: this is observed baseline evidence, not a transitive dependency lock.

Actual effective V5 `pip check` exited 1 with precisely three missing requirements:
`ima-racing 0.1.0` requires JupyterLab; CatBoost 1.2.10 requires Graphviz and Plotly.
Do not repair V5 or delete its metadata to conceal these failures. A private V6
clone must repair this closure. Actual `pip install --dry-run --no-cache-dir
--no-index` against both existing wheel stores and pinned locks exited 1 because
Pydantic 2.13.4 was unavailable. It requires pydantic-core 2.46.4; current base is
2.13.5/core 2.46.5. Strict locked V6 must install the pinned pair privately.

### Alternative Strict-Lock/Acquisition Recipe: Not Executed

These are additional requirements for the earlier proposed strict-lock/full
acquisition option, NOT permission to change the completed copy-mode identity.
Minimum additions for that option:

- Pydantic 2.13.4/core 2.46.4 to honor the research lock.
- JupyterLab, Graphviz, Plotly and their required closure for clean metadata.
  Local available candidate versions are 4.6.2, 0.21, 7.1.0 respectively, not
  remote installation receipts or newly approved transitive locks.
- Scrapy 2.19.0/pypdf 6.19.0 and transitives for full official acquisition/builds.
  Own acquisition wheels include Scrapy, lxml, parsel, Twisted, cryptography and
  pip; no pypdf wheel was found in the audited store. Do not copy an entire
  acquisition environment over the research stack.
- Playwright/browser only for separate existing UI acceptance, not headless
  continuous research. NumExpr, NGBoost, Ray and PySR are not required here.

Prepare missing Linux x86_64 CPython 3.12 wheels locally before transfer. Proposed
command (not run) resolves the added closure; retain a manifest and verify all
wheels on the real target rather than treating download success as acceptance:

```bash
V6_WHEELS=<fresh-local-wheel-directory>
.venv/bin/python -m pip download --only-binary=:all: --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 --implementation cp --python-version 3.12 --abi cp312 --dest "$V6_WHEELS" 'pydantic==2.13.4' 'pydantic-core==2.46.4' 'jupyterlab==4.6.2' 'graphviz==0.21' 'plotly==7.1.0' 'scrapy==2.19.0' 'pypdf==6.19.0'
scp -r "$V6_WHEELS" imaopt@100.95.24.121:/home/imaopt/research-v2/v6-dependencies/wheels
```

Prerequisites: parent committed release gates green; isolated own-user
`v6-dependencies` parent directory provisioned, with no `wheels` child before
the copy; fresh destination with disk
space for ~1.18 GB uncompressed copied packages plus source/data/wheels/canary;
complete hashed Linux wheel closure; verified frozen input/protocol; protected
baseline refreshed; explicit setup assignment. `scp` is a future mutation and
must not run during read-only preparation. Avoid accidental nested wheel paths:
destination must not exist before that copy.

The following own-user remote setup commands are prepared ONLY, not executed.
Use a fresh destination; preserve private permissions and stop if it exists.
Do not copy old venv `bin` scripts, use hardlinks, or change shared packages:

```bash
umask 077
V6_DEPS=/home/imaopt/research-v2/v6-dependencies
test ! -e "$V6_DEPS/venv"
test ! -e "$V6_DEPS/site-packages"
/home/imaopt/python/cpython-3.12.12/bin/python3.12 -m venv --without-pip "$V6_DEPS/venv"
cp -a /home/imaopt/research-v2/releases/2868bec/.venv/lib/python3.12/site-packages/. "$V6_DEPS/venv/lib/python3.12/site-packages/"
cp -a /home/imaopt/research-v2/v5-dependencies/site-packages "$V6_DEPS/site-packages"
export PYTHONPATH="$V6_DEPS/site-packages:$IMA_V6_RELEASE"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$V6_DEPS/venv/bin/python" -m pip install --no-cache-dir --no-index --find-links "$V6_DEPS/wheels" --find-links /home/imaopt/research-v2/wheelhouse --find-links /home/imaopt/research-v2/v5-dependencies/wheels --find-links /home/imaopt/acquisition/hkjc-20261002/wheels -r "$IMA_V6_RELEASE/requirements-research.lock" -r "$IMA_V6_RELEASE/requirements-v5-overlay.lock" 'jupyterlab==4.6.2' 'graphviz==0.21' 'plotly==7.1.0' 'scrapy==2.19.0' 'pypdf==6.19.0'
"$V6_DEPS/venv/bin/python" -m pip check
"$V6_DEPS/venv/bin/python" -m pip freeze --all
"$V6_DEPS/venv/bin/python" -m scripts.optimize run --help
"$V6_DEPS/venv/bin/python" -m scripts.run_research_expansion_canary --help
```

Never wildcard-install the old stores: they include NumPy 2.2.6 and Ray 2.49.0.
This uses copied pip 26.2.1 through the new interpreter; it does not need an
editable offline build or new setuptools/wheel bootstrapping. Copied project
metadata may reference old console-script paths; use verified `python -m`
entry points and source `PYTHONPATH`, never those stale scripts. Inspect
`sys.prefix`, `sys.path`, import origins and metadata locations after setup;
none may fall back to V5/acquisition package directories. Hash private files,
resolved dependency manifest and inputs; require `pip check` and isolated canary
before service installation. Offline resolution may expose additional gaps;
report them rather than silently enabling network or weakening locks.

Route/env refresh `2026-10-04T06:26:26Z` reconfirmed public models HTTP 200,
both OpenRouter addresses through `tailscale0` table 52, and own mode-0600
`/home/imaopt/.config/imaopt/openrouter.env` source with key present. No paid call
was made and no secret value or length displayed. Tailscale SSH transfer and
public egress are verified independently from authenticated model availability.

## Frozen Snapshot Bytes

The latest pointer at `2026-10-04T06:10:24.176781Z` still named
`/home/imaopt/acquisition/hkjc-20261002/snapshots/snapshot-20261004T050721Z-122062e8`.
Freeze that concrete path rather than resolving latest again during download.

| File | Exact bytes |
|---|---|
| `features.csv` | 353212946 |
| `features.parquet` | 53589138 |
| `runners.csv` | 121162762 |
| `runners.parquet` | 7496318 |
| `events.jsonl` | 138795437 |
| `profiles.jsonl` | 84965069 |
| `lineage.json` | 16029358 |
| `manifest.json` | 26891 |
| `exclusions.json` | 622 |
| `horse_seeds.json` | 803677 |
| Seven `code/` files, combined | 135121 |
| Full snapshot, all listed files | 776217339 |

Current V6 `DatasetFeatureProfile` and executor `_load_dataset` have Parquet
readers and the parent config selects `features.parquet`. Require end-to-end
preparation/canary readback of that
actual format, not just schema success. Transfer manifest, lineage, exclusions and code
provenance with whichever data format the parent chooses. The manifest records
SHA256 for the nine principal data/provenance files. Verify transferred bytes
against those hashes and freeze the manifest's own checksum separately. The
independent readback receipt is outside the snapshot, named
`snapshot-20261004T050721Z-122062e8.readback.json`, 2,767 bytes, and must be retained
too. The full snapshot plus that receipt totals 776,220,106 bytes.
Compression size is unmeasured; do not substitute the parquet size for the CSV
or whole-snapshot download requirement. No snapshot was downloaded by this role.

## V5 Graceful Handoff Procedure

This section is preparation only. Parent must explicitly assign the V5 safe
handoff after all V6 gates pass; no STOP or service mutation has been executed.
At inventory refresh V5 had no `STOP` marker.

Before any later assignment, save the protection digests, release alias/revision,
STOP absence, status/resource timestamps and own-unit invocation. The pinned
CLI writes a UTC timestamp to only the existing campaign's `STOP` file:

```bash
V5_CODE=/home/imaopt/research-v2/live-releases/ima-v5-current
V5_PYTHON=/home/imaopt/research-v2/releases/f54c604/.venv/bin/python
V5_CAMPAIGN=/home/imaopt/research-v2/campaigns/agentic_v5_discovery
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/imaopt/research-v2/v5-dependencies/site-packages:$V5_CODE "$V5_PYTHON" -m scripts.optimize stop --campaign "$V5_CAMPAIGN"
```

Pinned `research_v5.py:320` returns only after stop mode, no inflight workers,
planner completion, uploader completion and no pending tracking outbox when
tracking is enabled. Observe `draining`, admissions cessation, running count
reaching zero, pending tells/tracking zero and clean service exit. A STOP request
is not a stop-completion receipt, and an in-flight planner may finish after the
marker. Do not replace a slow drain with an unassigned service kill. A blocked
upload can prevent drain; retain evidence and report the actual blocker.

Authorized STOP creates a new marker hash and changes status/ledger/trials/
decision/evidence/outbox files as work drains; capture those new dynamic receipts.
It must not change the protected campaign identity, dataset/protocol, source,
config, unit definitions or dependency versions listed in validation. Add STOP
hash/UTC time to the handoff record. Any subsequent unexpected protected hash
change fails cutover verification. Never delete STOP automatically or rename V5
into V6; a later predecessor resume is its own assignment with original identity.

## Resource Envelope

The template sets `MemoryHigh=90000000000` and `MemoryMax=100000000000` bytes:
approximately 83.82 GiB high / 93.13 GiB maximum, leaving 10 decimal GB between
soft/hard thresholds. Systemd's `100G` is 100 GiB, 107,374,182,400 bytes, and is
not the approved 100 decimal GB envelope. Read effective byte values back with
`systemctl --user show` and cgroup files; template text alone is insufficient.

Current V5 is already using about 90-96 decimal GB in read-only samples. Running
another full envelope alongside it is unsafe. Preserve V5 while running only
an explicitly bounded small V6 canary with measured host headroom. Parent must
resolve any production handoff at an authorized safe boundary; this runbook does
not authorize stopping V5. Do not increase V6 concurrency until matched cold/warm
full-fold peaks, host reserve and other workloads support it. A 4 GiB scheduler
reservation is not a per-worker hard limit or a measured full-history fit size.
Record actual reserved and observed memory separately. Avoid `auto` initially.

Crash restart is limited to three starts per 900 seconds with a 60-second delay.
Clean policy/spend/STOP exits stay stopped. No V5 operational backoff hook is
attached to V6. V6 OOM/backoff/restart idempotency must be independently demonstrated
before unattended acceptance. `KillMode=control-group` limits forced shutdown to
the successor unit; `TimeoutStopSec=1800` is a last-resort bound, not proof of drain.

## Parent Launch And Readback

Only after gates pass, parent installs the own unit/env file, reloads the own user
manager and starts the successor. Do not enable a unit on reboot until recovery
and STOP behavior are verified. Record exact deployed source/data/protocol/config
and dependency hashes plus start time, invocation ID and service PID.

These probes are read-only:

```bash
systemctl --user show ima-research-expansion-supervisor.service -p ActiveState -p SubState -p MainPID -p NRestarts -p MemoryCurrent -p MemoryPeak -p MemoryHigh -p MemoryMax -p ControlGroup
journalctl --user -u ima-research-expansion-supervisor.service -n 40 --no-pager
curl --fail --silent http://100.95.24.121:5000/health
curl --fail --silent http://100.95.24.121:5000/version
```

Read `status.json` and the ledger together using a read-only SQLite connection
(`file:.../ledger.sqlite?mode=ro`); `scripts.optimize status` currently instantiates
`ResearchLedger` and is not promised to be strictly read-only. Compare UTC
timestamps, committed completions, fresh decision evidence, origin IDs and
tracking/tell watermarks. A service PID and healthy MLflow do not prove feedback.
Observe multiple real planning/execution windows: a paid plan while workers are
busy, meaningful typed action, completed composed graph, current comparable best,
model package replay and linked uncharged snapshot. Preserve API receipts and
existing MLflow UI screenshots for native USD/cost and run/model/dataset linkage.
Audit physical-call costs once, including failure/repair calls; unknown is null.

Recheck immutable V5 and acquisition-unit hashes after cutover. Dynamic V5
completions are expected to advance. Investigate unexpected identity/config/unit
changes immediately without touching other services. No blanket daemon/service
restart, shared dependency update, other-user repository access or firewall change.

## Successor Rollback

Rollback is parent-owned and must be tested as a dry run before acceptance.
Freeze the successor invocation/hashes, ledger/trials/decisions/evidence and
tracking watermark. Request graceful successor stop through the verified CLI:

```bash
"$IMA_V6_PYTHON" -m scripts.optimize stop --campaign "$IMA_V6_CAMPAIGN"
```

That command intentionally writes only the successor's `STOP` marker. Confirm
admission stops, active jobs drain, pending tells/tracking reconcile and the
successor process exits. If a failure prevents drain, parent may stop only
`ima-research-expansion-supervisor.service`; retain interrupted attempts and cause
evidence for reconciliation. Never delete V6 state, rewrite identities or reuse a
failed campaign with changed code/data. V5 remains pinned and running unless the
parent separately performed an authorized handoff; restoring a stopped predecessor
requires that explicit assignment and its original invocation, not an automatic
rollback command in this template. Keep evidence even when rollback succeeds.

## CI Contract

`.github/workflows/research-expansion.yml` runs on relevant pull requests, main and
this feature branch, or manual dispatch. Ubuntu 24.04/Python 3.12, existing extras
and lock files, `pip check`, resolved versions, Bash/systemd validation, CLI help
and `python -m unittest discover -s tests`. GitHub token has read-only contents;
checkout does not persist credentials; no remote deployment, API secrets or paid
planner calls. Cache keys include both locks and `pyproject.toml`, following the
[setup-python dependency-cache contract](https://github.com/actions/setup-python#caching-packages-dependencies).

Local passing tests and workflow syntax mean CI is prepared; only a real Actions
run can establish hosted-runner installation/test success. Full-history resource
benchmarks, historical source coverage, remote replay, MLflow native USD UI and
live recovery require dedicated evidence outside this fixture regression job.
