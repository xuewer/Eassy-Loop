"""Deterministic recomputation from archived scores, without model/API calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai_judge import parse_four, prompt_for

DIMENSIONS = ('relevance', 'grammar', 'structure', 'depth')
WEIGHTS = dict(zip(DIMENSIONS, (.3, .2, .2, .3)))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)), (a, b)


def total(scores):
    return sum(WEIGHTS[d] * scores[d] for d in DIMENSIONS)


def rank(values):
    return [1 + sum(v < x for v in values) + (sum(v == x for v in values) - 1) / 2 for x in values]


def agreement(pairs):
    if not pairs:
        return {'n': 0, 'mae': None, 'spearman': None}
    left, right = (rank(list(x)) for x in zip(*pairs))
    lm, rm = statistics.mean(left), statistics.mean(right)
    cov = sum((l-lm)*(r-rm) for l, r in zip(left, right))
    scale = math.sqrt(sum((l-lm)**2 for l in left) * sum((r-rm)**2 for r in right))
    return {'n': len(pairs), 'mae': statistics.mean(abs(a-b) for a, b in pairs),
            'spearman': cov/scale if scale else None}


def changes(values):
    interval = None
    if len(values) >= 2:
        rng = random.Random(0)
        means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
        interval = [means[50], means[1950]]
    return {'n': len(values), 'mean_delta': statistics.mean(values) if values else None,
            'improved': sum(x > 1e-12 for x in values), 'tied': sum(abs(x) <= 1e-12 for x in values),
            'regressed': sum(x < -1e-12 for x in values), 'ci95_sample_bootstrap': interval}


def check_changes(actual, expected):
    for key in ('n', 'improved', 'tied', 'regressed'):
        assert actual[key] == expected[key], (key, actual, expected)
    close(actual['mean_delta'], expected['mean_delta'])
    if expected['ci95_sample_bootstrap'] is None:
        assert actual['ci95_sample_bootstrap'] is None
    else:
        for a, b in zip(actual['ci95_sample_bootstrap'], expected['ci95_sample_bootstrap']):
            close(a, b)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Existing local dual-v2 archive; not included in the source distribution')
    parser.add_argument('--output', type=Path, help='Optional new verification record; archived files stay unchanged')
    parser.add_argument('--check-working-source', action='store_true', help='Also require the current source to match the archived experiment')
    args = parser.parse_args(argv)
    directory = args.run_dir.resolve()
    public_manifest = directory / 'MANIFEST.json'
    if public_manifest.exists():
        for name, expected in read(public_manifest)['files'].items():
            path = (directory / name).resolve()
            assert path.is_relative_to(directory), 'Manifest path escapes archive'
            assert digest(path) == expected, name
    policy = read(directory/'batch_plan.json')
    plan = read(directory/'ablation/plan.json')
    judge_plan = read(directory/'judge_plan.json')
    report = read(directory/'DUAL_MODEL_REPORT.json')
    state = read(directory/'status.json')
    assert state['status'] == 'completed'
    assert policy['human_review_required'] is False
    samples = {s['sample_id']: s for s in plan['samples']}
    assert len(samples) == 10
    assert digest(ROOT/'data/exploration_v1.jsonl') == plan['dataset_sha256']
    for name, expected in policy['provenance']['source_sha256'].items():
        if args.check_working_source:
            assert digest(ROOT/name) == expected
        assert digest(directory/'source_snapshot'/name) == expected
    index = [json.loads(s) for s in (directory/'ablation/index.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(index) == len({row['job_id'] for row in index}) == 40
    planned = {j['job_id']: j for j in plan['jobs']}
    jobs, first, second, original_second = {}, {}, {}, {}
    all_calls = [json.loads(s) for f in (directory/'journals').glob('*.jsonl') for s in f.read_text(encoding='utf-8').splitlines()]
    assert len(all_calls) == len({c['call_id'] for c in all_calls}) == state['calls_used'] == 320
    assert len(all_calls) <= policy['max_new_inference_attempts'] == 350
    calls_by_id = {c['call_id']: c for c in all_calls}
    for entry in index:
        jid = entry['job_id']; result = read(directory/'ablation'/Path(entry['result'].replace(chr(92), '/')))
        assert result['settings'] == planned[jid]['settings']
        assert len(result['calls']) <= planned[jid]['max_call_attempts']
        assert all(calls_by_id[c['call_id']] == c for c in result['calls'])
        assert result['history'][0]['essay'] == samples[entry['sample_id']]['essay']
        if result['final_score'] is not None:
            scores = {d: result[d+'_result']['judgment']['score'] for d in DIMENSIONS}
            close(total(scores), result['final_score'])
            first[jid] = {**scores, 'weighted_total': total(scores)}
        jobs[jid] = {**entry, 'result': result}
    judge_files = {f.stem: read(f) for f in (directory/'judge_results').glob('*.json')}
    assert len(judge_plan['items']) == len(judge_files) == 34
    for item in judge_plan['items']:
        row = judge_files[item['blind_id']]
        assert row['rater_type'] == 'ai' and row['human_attested'] is False
        assert row['essay_sha256'] == hashlib.sha256(item['essay'].encode()).hexdigest()
        call_rows = [json.loads(s) for s in (directory/'journals'/(item['blind_id']+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        assert len(call_rows) == 1
        assert call_rows[0]['prompt_sha256'] == hashlib.sha256(prompt_for(item).encode()).hexdigest()
        if row['status'] != 'ok':
            continue
        assert parse_four(row['raw_response'], item['essay']) == row['judgments']
        scores = {d: row['judgments'][d]['score'] for d in DIMENSIONS}
        scores['weighted_total'] = total(scores)
        for target in item['targets']:
            if target['kind'] == 'original': original_second[target['sample_id']] = scores
            else: second[target['job_id']] = scores
    assert report['workflow_attempted'] == 40 and report['judge_successful'] == 33
    assert report['successful_final_text_pairs'] == len(first.keys() & second.keys()) == 30
    for dimension in (*DIMENSIONS, 'weighted_total'):
        pairs, originals = [], []
        for sid in samples:
            local = [jid for jid in first.keys() & second.keys() if jobs[jid]['sample_id'] == sid]
            if local:
                pairs.append((statistics.mean(first[j][dimension] for j in local), statistics.mean(second[j][dimension] for j in local)))
            observations = []
            for job in jobs.values():
                if job['sample_id'] != sid: continue
                h = job['result']['history'][0]
                if dimension == 'weighted_total': value = h.get('final_score')
                else: value = next((d['judgment']['score'] for d in h['results'] if d['dimension'] == dimension and d['status'] == 'ok'), None)
                if value is not None: observations.append(value)
            if observations and sid in original_second:
                originals.append((statistics.mean(observations), original_second[sid][dimension]))
        for section, values in [('model_agreement_by_original_cluster_not_accuracy', pairs),
                                ('original_text_model_agreement_not_accuracy', originals)]:
            expected = agreement(values); actual = report[section][dimension]
            assert expected['n'] == actual['n']
            close(expected['mae'], actual['mae']); close(expected['spearman'], actual['spearman'])
    lookup = {(j['sample_id'], j['variant']): jid for jid, j in jobs.items()}
    for name, actual in report['comparisons'].items():
        left, right = name.split('_vs_'); avalues, bvalues = [], []
        failed = unrated = 0
        for sid in samples:
            a, b = lookup[sid, left], lookup[sid, right]
            if a not in first or b not in first: failed += 1
            elif a not in second or b not in second: unrated += 1
            else:
                avalues.append(first[a]['weighted_total']-first[b]['weighted_total'])
                bvalues.append(second[a]['weighted_total']-second[b]['weighted_total'])
        check_changes(actual['first_model'], changes(avalues)); check_changes(actual['second_model'], changes(bvalues))
        assert actual['failed_workflow_pairs'] == failed and actual['unrated_successful_pairs'] == unrated
        assert len(avalues)+failed+unrated+actual['missing_pairs'] == 10
    for variant, actual in report['variants'].items():
        local = [jid for jid, j in jobs.items() if j['variant'] == variant]
        successful = [jid for jid in local if jid in first]
        values = [second[j]['weighted_total']-original_second[jobs[j]['sample_id']]['weighted_total']
                  for j in successful if j in second and jobs[j]['sample_id'] in original_second]
        assert actual['workflow_success'] == len(successful)
        assert actual['workflow_failed'] == len(local)-len(successful)
        check_changes(actual['ai_changes_vs_original'], changes(values))
    manifest = {'verified_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
        'verification_type': 'deterministic_recomputation_not_independent_semantic_audit',
        'workflow_records': 40, 'workflow_success': len(first), 'judge_items': 34,
        'judge_success': 33, 'inference_attempts': len(all_calls), 'primary_paired_originals': 6,
        'checks': ['unique call IDs and total budget', 'source and snapshot hashes', 'dataset hash',
            'planned workflow settings and original text identity', 'all result calls match journal',
            'second-judge prompt hashes and raw JSON/evidence validation', 'weighted score arithmetic',
            'MAE and tied-rank Spearman recomputed', 'sample-level delta bootstrap intervals recomputed',
            'failed/unrated pair denominators', 'group change counts and negative outcomes'],
        'source_check': 'snapshot_and_working_tree' if args.check_working_source else 'archived_snapshot_only',
        'input_hashes': {f.relative_to(directory).as_posix(): digest(f) for f in [directory/'DUAL_MODEL_REPORT.json', directory/'batch_plan.json', directory/'judge_plan.json']},
        'new_model_calls': 0}
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print(json.dumps({k: manifest[k] for k in ('status','workflow_records','workflow_success','judge_items','inference_attempts','new_model_calls')}))


if __name__ == '__main__':
    main()
