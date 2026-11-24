"""Independent human annotation packets and paired-reference analysis."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics
from uuid import uuid4
from src.datasets import load_dataset
from src.metrics import agreement
from src.nodes import RUBRIC_VERSION, RUBRICS, ANCHORS
from src.schemas import DIMENSIONS, WEIGHTS


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def validate_scores(scores):
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        raise ValueError('All four human dimensions are required')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in scores.values()):
        raise ValueError('Human scores must be finite numbers in [0,1]; blanks are not labels')
    return scores


def absolute_icc(rows):
    """Two-way absolute-agreement single/average-rating ICC; balanced complete rows."""
    n = len(rows)
    if n < 3:
        return {'n': n, 'icc_a1': None, 'icc_a2': None, 'reason': 'Need at least three complete items'}
    k = 2
    grand = statistics.mean(x for row in rows for x in row)
    means = [statistics.mean(row) for row in rows]
    raters = [statistics.mean(row[j] for row in rows) for j in range(k)]
    msr = k * sum((x-grand)**2 for x in means)/(n-1)
    msc = n * sum((x-grand)**2 for x in raters)/(k-1)
    mse = sum((row[j]-means[i]-raters[j]+grand)**2 for i,row in enumerate(rows) for j in range(k))/((n-1)*(k-1))
    d1 = msr+(k-1)*mse+k*(msc-mse)/n
    d2 = msr+(msc-mse)/n
    return {'n':n, 'icc_a1':(msr-mse)/d1 if abs(d1)>1e-12 else None,
            'icc_a2':(msr-mse)/d2 if abs(d2)>1e-12 else None,
            'model':'Two-way absolute agreement; two fixed study raters, single and mean ratings; negative estimates retained'}


def agreement_with_ci(pairs, seed=42, repetitions=2000):
    result = agreement(pairs)
    if len(pairs) < 3:
        return {**result, 'mae_ci95':None, 'spearman_ci95':None, 'bootstrap_valid_spearman':0}
    rng = random.Random(seed)
    estimates = [agreement(rng.choices(pairs, k=len(pairs))) for _ in range(repetitions)]
    def interval(key):
        values = sorted(e[key] for e in estimates if e[key] is not None)
        return [values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]] if len(values)>=repetitions/2 else None
    return {**result, 'mae_ci95':interval('mae'), 'spearman_ci95':interval('spearman'),
            'bootstrap_valid_spearman':sum(e['spearman'] is not None for e in estimates),
            'ci_note':'Percentile bootstrap over items; n<=10 is exploratory, repeated model calls are not independent samples'}


def make_packet(items, destination, seed=42):
    """items contain private target keys; only topic, essay and random ID go to raters."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    mapping = {}
    rows = []
    for item in items:
        blind_id = 'h-'+uuid4().hex[:16]
        mapping[blind_id] = {k:v for k,v in item.items() if k not in ('topic','essay')}
        mapping[blind_id].update(essay_sha256=digest(item['essay']),topic_sha256=digest(item['topic']))
        rows.append({'blind_id':blind_id,'topic':item['topic'],'essay':item['essay'],
                     'rubric_version':RUBRIC_VERSION,'scores':dict.fromkeys(DIMENSIONS),
                     'notes':'','rater_id':'','rated_at':None,'human_attested':False,
                     'independent_of_model_scores':None})
    write_json(destination/'private_mapping.json',mapping)
    for rater in ('rater1','rater2'):
        order = rows.copy()
        random.Random(seed+(rater=='rater2')).shuffle(order)
        with (destination/(rater+'.jsonl')).open('x',encoding='utf-8') as f:
            for row in order: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        document = ['# Independent human annotation', '', 'Use an actual human rater ID. Do not inspect model scores, private mapping or the other rater.',
                    'Scores: relevance / grammar / structure / depth, each 0..1. Preserve every item ID and text.',
                    ANCHORS,'',*[d+': '+RUBRICS[d] for d in DIMENSIONS],'']
        for row in order:
            document.extend(['## '+row['blind_id'],row['topic'],'',row['essay'],'',
                'relevance: ____  grammar: ____  structure: ____  depth: ____',
                'Reason / disagreement notes: __________________________',''])
        (destination/(rater+'_reading.md')).write_text('\n'.join(document),encoding='utf-8')
        make_html(order,destination/(rater+'.html'),rater)
    write_json(destination/'packet_manifest.json',{'items':len(rows),'rubric_version':RUBRIC_VERSION,
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'human_labels':'NOT PROVIDED; templates are blank',
        'instructions':'Give each human only their html/jsonl and rubric. Keep private mapping and model scores hidden.'})
    return destination


def make_html(rows, path, name):
    data = json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    rubric = html.escape(ANCHORS+'\n'+'\n'.join(d+': '+RUBRICS[d] for d in DIMENSIONS))
    doc = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Eassy Loop independent human scoring</title>
<style>body{max-width:980px;margin:32px auto;padding:16px;font:17px/1.65 system-ui;color:#192535;background:#f5f7fa}article,header{background:white;padding:24px;margin-bottom:24px;border:1px solid #cdd5df;border-radius:10px}pre{white-space:pre-wrap;font:inherit}label{display:inline-block;margin:8px}input[type=number]{width:80px}textarea{width:98%;height:70px}button{padding:12px 24px;font:inherit}#message{white-space:pre-wrap;color:#a22929}</style>
<header><h1>独立人工评分 / Independent human scoring</h1><p>只填写真人独立判断。请勿查看模型分数、私有映射、其他标注者答案；不确定时说明原因，不用模型代评。</p>
<label>真实标注者 ID <input id="rater" required></label><p><label><input id="human" type="checkbox">我本人阅读文本并填写评分</label>
<label><input id="independent" type="checkbox">评分时未查看模型评分或其他标注者分数</label></p>
<p>未满足独立性可保留未勾选并在备注说明；报告会记录这一限制。数值为 0–1，建议以 0.05 为步长。</p><pre>__RUBRIC__</pre>
<p>页面不会联网，也不自动保存。可随时导出草稿；关闭前务必导出。原始草稿不能用于统计。</p><button id="draft">导出草稿</button> <button id="export">校验并导出最终 JSONL</button><p id="message"></p></header><main id="items"></main>
<script>const rows=__ROWS__;const dims=['relevance','grammar','structure','depth'];
for(const row of rows){const a=document.createElement('article');const h=document.createElement('h2');h.textContent=row.blind_id;const t=document.createElement('h3');t.textContent=row.topic;const e=document.createElement('pre');e.textContent=row.essay;a.append(h,t,e);for(const d of dims){const l=document.createElement('label');l.textContent=d+' ';const s=document.createElement('input');s.type='number';s.min=0;s.max=1;s.step=.05;s.id=row.blind_id+'-'+d;l.append(s);a.append(l)}const n=document.createElement('textarea');n.id=row.blind_id+'-notes';n.placeholder='评分理由、证据与不确定性（不得修改原文）';a.append(document.createElement('br'),n);document.querySelector('#items').append(a)}
function save(final){try{const who=document.querySelector('#rater').value.trim();if(final&&(!who||!document.querySelector('#human').checked))throw Error('填写真实标注者 ID 并确认本人评分。');const output=rows.map(row=>{const scores={};for(const d of dims){const v=document.getElementById(row.blind_id+'-'+d).value;scores[d]=v===''?null:Number(v);if(final&&(scores[d]===null||!Number.isFinite(scores[d])||scores[d]<0||scores[d]>1))throw Error(row.blind_id+' '+d+' 需要0–1数值')}return {...row,scores,notes:document.getElementById(row.blind_id+'-notes').value,rater_id:who,rated_at:new Date().toISOString(),human_attested:document.querySelector('#human').checked,independent_of_model_scores:document.querySelector('#independent').checked}});const url=URL.createObjectURL(new Blob([output.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/x-ndjson;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download=(final?'__NAME___completed':'__NAME___draft')+'.jsonl';link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);document.querySelector('#message').textContent='已导出。请保留原始评分，不查看另一位标注者答案。'}catch(e){document.querySelector('#message').textContent=e.message}}document.querySelector('#draft').onclick=()=>save(false);document.querySelector('#export').onclick=()=>save(true);</script></html>'''
    Path(path).write_text(doc.replace('__RUBRIC__',rubric).replace('__ROWS__',data).replace('__NAME__',name),encoding='utf-8')


def read_labels(path,mapping):
    rows = [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    labels, raters = {}, set()
    for row in rows:
        key = row.get('blind_id')
        if key in labels or key not in mapping:
            raise ValueError('Duplicate or unknown blind ID')
        ref = mapping[key]
        if digest(row.get('essay','')) != ref['essay_sha256'] or digest(row.get('topic','')) != ref['topic_sha256']:
            raise ValueError('Annotation text changed')
        if row.get('rubric_version') != RUBRIC_VERSION:
            raise ValueError('Rubric differs')
        if row.get('human_attested') is not True or not isinstance(row.get('rater_id'),str) or not row['rater_id'].strip():
            raise ValueError('Completed labels require actual human rater ID and attestation; blank templates are not results')
        if type(row.get('independent_of_model_scores')) is not bool:
            raise ValueError('Record whether the rater was blind to model scores')
        if not isinstance(row.get('rated_at'),str):
            raise ValueError('Human annotation timestamp is required')
        datetime.fromisoformat(row['rated_at'].replace('Z','+00:00'))
        validate_scores(row.get('scores'))
        raters.add(row['rater_id'])
        labels[key] = row
    if len(raters)!=1:
        raise ValueError('One nonempty rater identity per annotation file is required')
    return labels, next(iter(raters))


def combine_labels(paths,mapping):
    if not 1 <= len(paths) <= 2:
        raise ValueError('Provide one or two human annotation files')
    loaded = [read_labels(p,mapping) for p in paths]
    if len({who for _,who in loaded}) != len(loaded):
        raise ValueError('Two raters must have distinct identities')
    common = set.intersection(*(set(rows) for rows,_ in loaded))
    combined = {key:{d:statistics.mean(rows[key]['scores'][d] for rows,_ in loaded) for d in DIMENSIONS} for key in common}
    disagreements, inter = [], {}
    if len(loaded)==2:
        a,b = loaded[0][0], loaded[1][0]
        for key in sorted(common):
            disagreements.append({'blind_id':key,**mapping[key],
                'rater1_scores':a[key]['scores'],'rater2_scores':b[key]['scores'],
                'absolute_differences':{d:abs(a[key]['scores'][d]-b[key]['scores'][d]) for d in DIMENSIONS},
                'rater1_notes':a[key].get('notes',''),'rater2_notes':b[key].get('notes','')})
        for d in (*DIMENSIONS,'weighted_total'):
            def val(row): return sum(WEIGHTS[x]*row['scores'][x] for x in DIMENSIONS) if d=='weighted_total' else row['scores'][d]
            pairs = [(val(a[k]),val(b[k])) for k in sorted(common)]
            inter[d] = {**agreement(pairs),**absolute_icc(pairs),
                        'ci_note':'No inter-rater CI: variant texts can share an initial sample; do not treat them as independent items'}
    return combined, {'raters':[who for _,who in loaded],'rated_items_per_rater':[len(rows) for rows,_ in loaded],
        'common_items':len(common),'reference_policy':'single human' if len(loaded)==1 else 'mean of two human scores on commonly rated items; raw disagreements preserved',
        'all_raters_reported_blind':all(row['independent_of_model_scores'] for rows,_ in loaded for row in rows.values()),
        'inter_rater':inter or None,'disagreements':disagreements,
        'limitation':'AI-authored n=10 convenience set; not validation of a complete evaluation system. Attestation records declarations, not verified rater independence.'}


def analyze_reference(run_dir,packet,ratings):
    from src.experiments import load_jobs
    plan,jobs = load_jobs(run_dir)
    if plan['kind']!='reliability': raise ValueError('Reference analysis requires reliability runs')
    mapping = json.loads((Path(packet)/'private_mapping.json').read_text(encoding='utf-8'))
    refs,metadata = combine_labels(ratings,mapping)
    by_sample = {v['sample_id']:refs[k] for k,v in mapping.items() if k in refs}
    for v in mapping.values():
        matches = [s for s in plan['samples'] if s['sample_id']==v['sample_id']]
        if len(matches)!=1 or digest(matches[0]['essay'])!=v['essay_sha256'] or digest(matches[0]['topic'])!=v['topic_sha256']:
            raise ValueError('Reference packet differs from frozen dataset')
    predictions = {}
    for sample in plan['samples']:
        subset = [e['result'] for e in jobs if e['sample_id']==sample['sample_id']]
        dims = {}
        for d in DIMENSIONS:
            values = [(r.get(d+'_result') or {}).get('judgment',{}).get('score') for r in subset if (r.get(d+'_result') or {}).get('status')=='ok']
            dims[d] = statistics.mean(values) if values else None
        totals = [r['final_score'] for r in subset if r.get('final_score') is not None]
        predictions[sample['sample_id']] = {**dims,'weighted_total':statistics.mean(totals) if totals else None}
    metrics = {}
    for d in (*DIMENSIONS,'weighted_total'):
        pairs = []
        for key,ref in by_sample.items():
            value = predictions.get(key,{}).get(d)
            if value is not None:
                human = sum(WEIGHTS[x]*ref[x] for x in DIMENSIONS) if d=='weighted_total' else ref[d]
                pairs.append((human,value))
        metrics[d] = agreement_with_ci(pairs)
    return {'human_annotation':metadata,'model_vs_human':metrics,'model_repeated_mean_predictions':predictions,
            'warning':'Repeated model ratings averaged per essay before comparison; n=10 exploratory only. No labels are generated by this command.'}


def analyze_ablation(run_dir,packet,ratings):
    from src.experiments import analyze, load_jobs
    plan,jobs = load_jobs(run_dir)
    if plan['kind']!='ablation': raise ValueError('Requires ablation experiment')
    mapping = json.loads((Path(packet)/'private_mapping.json').read_text(encoding='utf-8'))
    refs,metadata = combine_labels(ratings,mapping)
    by_job = {e['job_id']:e for e in jobs}
    annotations = {}
    for key,value in mapping.items():
        for job_id in value.get('job_ids',[value.get('job_id')]):
            if job_id not in by_job: raise ValueError('Unknown mapped job')
            result = by_job[job_id]['result']
            if digest(result['essay'])!=value['essay_sha256'] or digest(result['topic'])!=value['topic_sha256']:
                raise ValueError('Blind packet differs from result')
            if key in refs: annotations[job_id] = sum(WEIGHTS[d]*refs[key][d] for d in DIMENSIONS)
    report = analyze(run_dir, human_job_scores=annotations)
    return {**report,'human_annotation':metadata,
        'primary_comparison':'all_feedback_vs_no_feedback',
        'claim_limit':'Human score deltas on ten AI-authored exploratory samples; one rewrite realization per condition. Negative results retained, no system-wide validation claim.'}


def main(argv=None):
    p=argparse.ArgumentParser(description='Human reference and ablation analysis; never generates human labels')
    sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export-reference');e.add_argument('--dataset',type=Path,required=True);e.add_argument('--output-dir',type=Path,required=True)
    e=sub.add_parser('export-ablation');e.add_argument('--run-dir',type=Path,required=True);e.add_argument('--output-dir',type=Path,required=True)
    for name in ('analyze-reference','analyze-ablation'):
        a=sub.add_parser(name);a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--packet',type=Path,required=True)
        a.add_argument('--ratings',type=Path,nargs='+',required=True);a.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='export-reference':
            samples,_=load_dataset(a.dataset)
            make_packet([{k:s[k] for k in ('sample_id','topic','essay')} for s in samples],a.output_dir)
        elif a.command=='export-ablation':
            from src.experiments import load_jobs
            _,jobs=load_jobs(a.run_dir)
            grouped={}
            for e in jobs:
                r=e['result']
                if not r.get('essay'): continue
                # Preserve texts even when final evaluation failed; failures stay explicit in analysis.
                key=(r['topic'],r['essay'])
                grouped.setdefault(key,[]).append(e['job_id'])
            make_packet([{'topic':t,'essay':s,'job_ids':ids} for (t,s),ids in grouped.items()],a.output_dir)
        else:
            result=(analyze_reference if a.command=='analyze-reference' else analyze_ablation)(a.run_dir,a.packet,a.ratings)
            a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,result)
        print(json.dumps({'status':'completed','note':'Human scores must be supplied by actual raters; no AI-generated reference labels.'}))
    except (ValueError,KeyError,TypeError,OSError):
        print(json.dumps({'status':'input_error','hint':'Check packet identity, score completeness, actual human attestation, and new output path.'}))
        return 2
    return 0


if __name__=='__main__': raise SystemExit(main())
