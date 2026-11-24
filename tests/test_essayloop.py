import json
from pathlib import Path
import pytest
import httpx
from src.app import main, run
from src.artifacts import save_result, summary
from src.config import Settings
from src.costs import calculate_cost
from src.datasets import load_dataset
from src.demo import DemoGenerator, DemoEvaluator
from src.experiments import make_plan, execute_plan, analyze, export_blind
from src.metrics import agreement, bootstrap_mean_interval, ranks


def cfg(**kw):
    return Settings(retry_delay_seconds=0, **kw)


def demo_run(**kw):
    return run('AI', generator=DemoGenerator(), evaluator=DemoEvaluator(), **kw)


@pytest.mark.parametrize('execution', ['serial', 'parallel'])
def test_real_graph_closed_loop_with_injected_models(execution):
    result = demo_run(settings=cfg(execution=execution))
    assert result['exit_reason'] == 'threshold_met'
    assert result['rewrites_completed'] == 1
    assert len(result['history']) == 2
    assert len(result['generation_history']) == 2
    assert result['generation_history'][1]['feedback_used'] == result['history'][0]['feedback']
    assert result['generation_history'][1]['input_essay_hash'] != result['generation_history'][1]['output_essay_hash']
    assert len(result['calls']) == 10
    assert all(c['latency_seconds'] >= 0 for c in result['calls'])
    assert result['telemetry']['usage_complete'] is False
    assert result['telemetry']['total_tokens'] is None
    assert result['rubric_version'] == 'essayloop-rubric-v3'


@pytest.mark.parametrize('feedback_mode,dimensions', [('all',4),('relevance',1),('none',0)])
def test_feedback_ablation_reaches_rewriter(feedback_mode, dimensions):
    result = demo_run(settings=cfg(feedback_mode=feedback_mode, stop_policy='fixed', max_rounds=2))
    feedback = result['generation_history'][1]['feedback_used']
    assert len(json.loads(feedback)) == dimensions if dimensions else feedback == ''
    assert result['exit_reason'] == 'max_rounds'


def test_fixed_policy_prevents_early_exit():
    result = demo_run(settings=cfg(stop_policy='fixed', threshold=.1, max_rounds=3))
    assert len(result['history']) == 3
    assert result['rewrites_completed'] == 2


def test_dimension_floor_is_explicit():
    result = demo_run(settings=cfg(threshold=.4, min_dimension_score=.8, max_rounds=1))
    assert result['exit_reason'] == 'max_rounds'


def test_artifacts_are_complete_and_never_overwritten(tmp_path):
    result = demo_run(settings=cfg(max_rounds=1))
    output = tmp_path/'run'
    path = save_result(result, output)
    assert json.loads(path.read_text(encoding='utf-8'))['essay'] == result['essay']
    assert len((output/'calls.jsonl').read_text().splitlines()) == 5
    assert json.loads((output/'summary.json').read_text())['telemetry']['call_count'] == 5
    with pytest.raises(FileExistsError):
        save_result(result, output)


def test_cli_demo_is_explicit_and_offline(tmp_path):
    output=tmp_path/'cli'
    assert main(['--demo','--output-dir',str(output)]) == 0
    result=json.loads((output/'result.json').read_text())
    assert result['evidence_type']=='scripted_demo_not_experiment'
    assert json.loads((output/'summary.json').read_text())['evidence_type']=='scripted_demo_not_experiment'
    assert main(['--demo','--output-dir',str(output)]) == 2


def dataset(tmp_path, scores=None):
    path=tmp_path/'samples.jsonl'
    sample={'sample_id':'s1','topic':'AI','essay':'AI changes work.', 'split':'dev','source':'test fixture, not a benchmark','human_scores':scores}
    path.write_text(json.dumps(sample)+'\n',encoding='utf-8')
    return path


def test_duplicate_samples_rejected(tmp_path):
    p=dataset(tmp_path)
    p.write_text(p.read_text()*2)
    with pytest.raises(ValueError):load_dataset(p)


@pytest.mark.parametrize('kind,jobs', [('reliability',2),('efficiency',4),('ablation',8),('early_exit',4)])
def test_reproducible_plans_and_call_bounds(tmp_path, kind, jobs):
    path=dataset(tmp_path)
    a=make_plan(path,kind,cfg(max_rounds=2),repeats=2)
    b=make_plan(path,kind,cfg(max_rounds=2),repeats=2)
    assert a['jobs']==b['jobs']
    assert len(a['jobs'])==jobs
    assert a['max_call_attempts']==sum(j['max_call_attempts'] for j in a['jobs'])
    for j in a['jobs']:
        assert j['sample_id']=='s1'


def test_live_execution_and_budget_guards_before_calls(tmp_path):
    p=make_plan(dataset(tmp_path),'efficiency',cfg(),repeats=1)
    with pytest.raises(ValueError,match='execute'):
        execute_plan(p,tmp_path/'run',max_calls=100)
    with pytest.raises(ValueError,match='max_calls'):
        execute_plan(p,tmp_path/'run',demo=True,max_calls=0)
    assert not (tmp_path/'run').exists()


def test_offline_experiment_analysis_and_blind_export(tmp_path):
    p=make_plan(dataset(tmp_path),'ablation',cfg(max_rounds=2),repeats=1)
    directory=tmp_path/'run'
    execute_plan(p,directory,demo=True,max_calls=p['max_call_attempts'])
    report=analyze(directory)
    assert report['completed_jobs']==4
    assert report['evidence_types']==['scripted_demo_not_experiment']
    assert report['comparisons']['all_feedback_vs_no_rewrite']['mean_delta'] is None
    exported=export_blind(directory,tmp_path/'blind')
    assert exported['annotation_count']==4
    rows=[json.loads(l) for l in (tmp_path/'blind'/'annotation.jsonl').read_text().splitlines()]
    assert all('variant' not in row and 'sample_id' not in row for row in rows)
    for row in rows:row['scores']={d:.7 for d in ('relevance','grammar','structure','depth')}
    labels=tmp_path/'filled.jsonl'
    labels.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    labeled=analyze(directory,annotation_path=labels,mapping_path=tmp_path/'blind'/'private_mapping.json')
    assert labeled['comparisons']['all_feedback_vs_no_rewrite']['mean_delta']==pytest.approx(0)


def test_reliability_without_labels_never_invents_accuracy(tmp_path):
    p=make_plan(dataset(tmp_path),'reliability',cfg(),repeats=2)
    execute_plan(p,tmp_path/'run',demo=True,max_calls=p['max_call_attempts'])
    report=analyze(tmp_path/'run')
    for dim in report['reliability']['dimensions'].values():
        assert dim['n']==0 and dim['mae'] is None and dim['spearman'] is None
        assert dim['samples_with_repeats']==1


def test_demo_judge_quotes_each_actual_dataset_essay(tmp_path):
    from src.experiments import make_plan, execute_plan, analyze
    p=make_plan(Path(__file__).resolve().parents[1]/'data'/'examples.jsonl', 'ablation',
                cfg(max_rounds=2, max_attempts=1), repeats=1)
    execute_plan(p,tmp_path/'demo',demo=True,max_calls=p['max_call_attempts'])
    report=analyze(tmp_path/'demo')
    assert report['complete'] is True
    assert all(v['successful']==v['planned'] for v in report['variants'].values())


def test_metrics_ties_and_small_sample():
    assert ranks([2,1,2])==[2.5,1,2.5]
    assert agreement([(.2,.2),(.8,.8)])['spearman']==pytest.approx(1)
    assert agreement([(.5,.5),(.5,.8)])['spearman'] is None
    assert bootstrap_mean_interval([1]) is None
    assert bootstrap_mean_interval([1,2], repetitions=100)==bootstrap_mean_interval([1,2], repetitions=100)


def test_costs_require_known_usage_and_prices():
    prices={'currency':'TEST','as_of':'2026-11-08','source':'synthetic unit test rates; not provider prices',
            'models':{'fake':{'input_per_million':2,'output_per_million':4}}}
    c=[{'model':'fake','input_tokens':100,'output_tokens':20}]
    assert calculate_cost(c,prices)['complete_cost']==pytest.approx(.00028)
    c.append({'model':'fake','input_tokens':None,'output_tokens':None})
    assert calculate_cost(c,prices)['complete_cost'] is None
    assert calculate_cost(c,prices)['calls_unpriced']==1


def test_provider_usage_not_guessed_and_thread_metadata_cleared(monkeypatch):
    import src.client as module
    monkeypatch.setattr(module,'load_environment',lambda:None)
    monkeypatch.setenv('DASHSCOPE_API_KEY','offline-test')
    client=module.QwenClient(cfg())
    client.http.close()
    client.http=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={
        'choices':[{'finish_reason':'stop','message':{'content':'essay'}}],
        'usage':{'prompt_tokens':12,'completion_tokens':5}})))
    try:
        assert client.invoke('test')=='essay'
        assert client.take_metadata()['input_tokens']==12
        assert client.take_metadata()=={}
    finally:client.close()
