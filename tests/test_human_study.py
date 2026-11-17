"""All ratings below are synthetic unit fixtures, not human study results."""
import json
from pathlib import Path
import pytest
from src.annotation import (absolute_icc, agreement_with_ci, make_packet, read_labels,
    combine_labels, analyze_reference, analyze_ablation, main)
from src.config import Settings
from src.datasets import load_dataset
from src.experiments import make_plan, execute_plan, analyze
from src.schemas import DIMENSIONS


def test_icc_perfect_and_systematic_rater_bias():
    assert absolute_icc([(.1,.1),(.5,.5),(.9,.9)])['icc_a1']==pytest.approx(1)
    assert absolute_icc([(.1,.3),(.4,.6),(.7,.9)])['icc_a1'] < 1
    assert absolute_icc([(.5,.5)]*3)['icc_a1'] is None
    assert absolute_icc([(.1,.1)])['icc_a1'] is None


def test_bootstrap_metrics_and_constants():
    r=agreement_with_ci([(.1,.1),(.4,.4),(.8,.8)],repetitions=100)
    assert r['mae']==0 and r['spearman']==pytest.approx(1)
    assert r['mae_ci95']==[0,0]
    assert agreement_with_ci([(.5,.2)]*3,repetitions=100)['spearman'] is None


def dataset(tmp_path):
    p=tmp_path/'dataset.jsonl'
    rows=[{'sample_id':f's{i}','topic':'What makes a good review?','essay':f'A review explains its limits. Example {i}.',
           'split':'dev','source':'synthetic test fixture, not study evidence'} for i in range(3)]
    p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
    return p,rows


def fill(packet, rater, value=.5):
    p=packet/(rater+'.jsonl')
    rows=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
    for i,row in enumerate(rows):
        row.update(scores=dict.fromkeys(DIMENSIONS,value),rater_id=rater,
            human_attested=True,independent_of_model_scores=True,rated_at='2026-11-13T00:00:00+00:00',notes='Synthetic unit test fixture only')
    target=packet/(rater+'_testfixture.jsonl')
    target.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    return target


def test_packet_has_no_model_scores_or_quality_stratum_and_rejects_blank(tmp_path):
    _,rows=dataset(tmp_path)
    packet=make_packet([{k:s[k] for k in ('sample_id','topic','essay')} for s in rows],tmp_path/'packet')
    mapping=json.loads((packet/'private_mapping.json').read_text())
    with pytest.raises(ValueError,match='human'):
        read_labels(packet/'rater1.jsonl',mapping)
    content=(packet/'rater1.html').read_text(encoding='utf-8')
    assert 'sample_id' not in content and 'final_score' not in content
    assert ".join('\\n')" in content
    a,b=fill(packet,'rater1',.4),fill(packet,'rater2',.6)
    refs,metadata=combine_labels([a,b],mapping)
    assert len(refs)==3 and metadata['common_items']==3
    assert all(ref['depth']==pytest.approx(.5) for ref in refs.values())
    assert metadata['inter_rater']['depth']['mae']==pytest.approx(.2)
    assert len(metadata['disagreements'])==3
    with pytest.raises(ValueError,match='distinct'): combine_labels([a,a],mapping)


def test_reference_and_ablation_end_to_end_without_real_labels(tmp_path):
    path,rows=dataset(tmp_path)
    config=Settings(max_rounds=2,max_attempts=1,retry_delay_seconds=0)
    plan=make_plan(path,'reliability',config,repeats=3)
    run_dir=tmp_path/'reliability'
    execute_plan(plan,run_dir,demo=True,max_calls=plan['max_call_attempts'])
    report=analyze(run_dir)
    assert report['reliability']['failed_jobs']==0
    assert len(report['reliability']['per_sample'])==3
    for row in report['reliability']['per_sample']:
        assert row['total_score']['range']==0
        assert row['dimensions']['depth']['sample_sd']==0
        assert row['successful_repeats']==3
    packet=make_packet([{k:s[k] for k in ('sample_id','topic','essay')} for s in rows],tmp_path/'refs')
    reference=analyze_reference(run_dir,packet,[fill(packet,'rater1')])
    assert reference['model_vs_human']['depth']['mae']==pytest.approx(0)
    assert reference['human_annotation']['inter_rater'] is None
    p=make_plan(path,'ablation',config,repeats=1)
    out=tmp_path/'ablation'
    execute_plan(p,out,demo=True,max_calls=p['max_call_attempts'])
    blind=tmp_path/'ablation-blind'
    assert main(['export-ablation','--run-dir',str(out),'--output-dir',str(blind)])==0
    result=analyze_ablation(out,blind,[fill(blind,'rater1')])
    c=result['comparisons']['all_feedback_vs_no_feedback']
    assert c['tied_count']==3 and c['improved_count']==c['regressed_count']==0
    assert result['primary_comparison']=='all_feedback_vs_no_feedback'
    assert result['evidence_types']==['scripted_demo_not_experiment']


def test_ten_frozen_texts_have_unique_topics_no_fake_labels():
    samples,_=load_dataset(Path(__file__).resolve().parents[1]/'data'/'exploration_v1.jsonl')
    assert len(samples)==10 and len({s['topic'] for s in samples})==10
    assert all(s['human_scores'] is None for s in samples)
    assert all(280 <= len(s['essay'].split()) <= 520 for s in samples)


def test_study_freezes_expected_budget_and_does_not_execute(tmp_path):
    from src.study import prepare, run_study
    p=prepare(tmp_path/'study',model='offline-test')
    r=json.loads((p/'reliability_plan.json').read_text())
    a=json.loads((p/'ablation_plan.json').read_text())
    assert len(r['jobs'])==30 and r['max_call_attempts']==120
    assert len(a['jobs'])==40 and a['max_call_attempts']==310
    assert all(j['settings']['max_rounds']==2 for j in a['jobs'])
    assert (p/'human_reference'/'rater1.html').exists()
    with pytest.raises(ValueError,match='execute'):run_study(p,max_calls=430)
    assert not (p/'reliability').exists()
