"""Export an allowlisted experiment bundle without local credentials or review logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

FILES = (
    'batch_plan.json', 'judge_plan.json', 'status.json', 'DUAL_MODEL_REPORT.json',
    'SCORE_COMPARISON.jsonl', 'DETERMINISTIC_VERIFICATION.json', 'ablation/plan.json',
    'ablation/index.jsonl', 'verification_20261117/pytest.xml',
)


def export(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError('Choose a new output directory')
    selected = [source / name for name in FILES]
    selected += sorted((source / 'ablation').glob('job-*/result.json'))
    selected += sorted((source / 'journals').glob('*.jsonl'))
    selected += sorted((source / 'judge_results').glob('*.json'))
    selected += sorted((source / 'source_snapshot/src').glob('*.py'))
    selected += [source / 'source_snapshot' / name for name in
                 ('requirements.txt', 'requirements-lock.txt')]
    if not all(path.is_file() for path in selected):
        raise ValueError('Experiment archive is incomplete')
    destination.mkdir(parents=True)
    hashes = {}
    for path in selected:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'format': 'byte_preserving_public_evidence_v1',
        'source_batch': source.name,
        'files': hashes,
        'note': 'Raw records and historical source names are unchanged. Archived local project paths may appear in provenance. Credentials, environment files and assistant review transcripts are not included.',
    }
    (destination / 'MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'files': len(hashes), 'output': str(destination), 'new_model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir, args.output_dir)))


if __name__ == '__main__':
    main()
