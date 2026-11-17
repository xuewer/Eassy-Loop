"""Synthetic fixtures only; no human or model quality evidence."""
import json
from pathlib import Path
import pytest
from src.ai_judge import prepare, execute, parse_four, prompt_for
from src.config import Settings
from src.experiments import make_plan,execute_plan
from src.schemas import DIMENSIONS
from src.client import ProviderError


class FakeSecondJudge:
    def __init__(self):self.calls=0
    def invoke(self,prompt):
        self.calls+=1
        data=json.loads(prompt.split('INPUT_JSON:\n')[1])
        quote=data['essay'][:12]
        return json.dumps({'judgments':[{'dimension':d,'score':.5,'rationale':'Synthetic test only',
            'evidence':[quote],'suggestions':['Review the reasoning']} for d in DIMENSIONS]})


def study(tmp_path):
    root=tmp_path/'study';root.mkdir()
    dataset=tmp_path/'dataset.jsonl'
    dataset.write_text(json.dumps({'sample_id':'s1','topic':'AI','essay':'AI changes work.',
        'split':'dev','source':'synthetic unit fixture only'})+'\n',encoding='utf-8')
    settings=Settings(max_rounds=2,max_attempts=1,retry_delay_seconds=0)
    for kind in ('reliability','ablation'):
        p=make_plan(dataset,kind,settings,repeats=1)
        execute_plan(p,root/kind,demo=True,max_calls=p['max_call_attempts'])
    return root


def test_prompt_hides_variants_scores_ids_and_quality_profiles():
    p=prompt_for({'topic':'A topic','essay':'A text.','targets':[{'variant':'PRIVATE_VARIANT'}],
        'prior_score':'PRIVATE_SCORE','sample_id':'PRIVATE_ID'})
    assert 'PRIVATE_' not in p


def test_parser_rejects_duplicate_dimensions_and_fake_quotes():
    judge=FakeSecondJudge()
    raw=judge.invoke(prompt_for({'topic':'AI','essay':'AI changes work.'}))
    assert len(parse_four(raw,'AI changes work.'))==4
    data=json.loads(raw);data['judgments'][1]['dimension']='relevance'
    with pytest.raises(ValueError):parse_four(json.dumps(data),'AI changes work.')
    data=json.loads(raw);data['judgments'][0]['evidence']=['Not in essay']
    with pytest.raises(ValueError):parse_four(json.dumps(data),'AI changes work.')


def test_ai_ratings_are_separate_and_never_human_attested(tmp_path,monkeypatch):
    monkeypatch.setenv('QWEN_BASE_URL','https://example.invalid/v1')
    root=study(tmp_path);out=tmp_path/'judge'
    plan=prepare(root,out,'kimi-k3')
    assert plan['planned_calls']==2  # Repeated unchanged original and rewrite deduplicated.
    fake=FakeSecondJudge()
    report=execute(out,execute=True,max_calls=50,model=fake)
    assert fake.calls==2
    assert report['successful_ai_ratings']==2
    assert report['human_reference_metrics'] is None
    assert report['human_inter_rater_agreement'] is None
    for line in (out/'ratings.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        assert row['rater_type']=='ai' and row['human_attested'] is False
    assert report['primary_ai_comparison_all_feedback_vs_no_feedback']['paired_samples']==1
    with pytest.raises(ValueError,match='Already started'):execute(out,execute=True,max_calls=50,model=fake)
    assert fake.calls==2


def test_provider_failure_stops_after_one_attempt_and_does_not_fallback(tmp_path,monkeypatch):
    monkeypatch.setenv('QWEN_BASE_URL','https://example.invalid/v1')
    root=study(tmp_path);out=tmp_path/'judge';prepare(root,out,'kimi-k3')
    class Denied:
        def __init__(self):self.calls=0
        def invoke(self,prompt):
            self.calls+=1
            raise ProviderError('DO_NOT_LOG_SECRET',category='not_found',http_status=404)
    fake=Denied()
    report=execute(out,execute=True,max_calls=50,model=fake)
    assert fake.calls==1 and report['provider_failures']==1
    assert report['unattempted']==1 and report['successful_ai_ratings']==0
    assert 'DO_NOT_LOG_SECRET' not in (out/'ratings.jsonl').read_text()
    assert json.loads((out/'completion.json').read_text())['status']=='stopped_on_provider_error'
