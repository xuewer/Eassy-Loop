"""Bounded fixed n=10 exploration; human labeling remains an external human task."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from src.annotation import make_packet, main as annotation_main
from src.config import PROJECT_ROOT, Settings
from src.datasets import load_dataset
from src.experiments import make_plan, execute_plan, analyze, dump, load_jobs
from src.plans import validate_plan


def prepare(destination, model='qwen3.8-27b'):
    destination=Path(destination)
    dataset=PROJECT_ROOT/'data'/'exploration_v1.jsonl'
    samples,digest=load_dataset(dataset)
    if len(samples)!=10: raise ValueError('Frozen study requires exactly ten essays')
    settings=Settings(model=model,max_rounds=2,max_attempts=1,timeout_seconds=180,
        retry_delay_seconds=0,stop_policy='fixed',execution='parallel',feedback_mode='all')
    plans={kind:make_plan(dataset,kind,settings,repeats=3 if kind=='reliability' else 1,seed=20261113)
           for kind in ('reliability','ablation')}
    for plan in plans.values():validate_plan(plan)
    if sum(p['max_call_attempts'] for p in plans.values())!=430:raise ValueError('Unexpected study budget')
    destination.mkdir(parents=True,exist_ok=False)
    for kind,plan in plans.items():dump(destination/(kind+'_plan.json'),plan)
    make_packet([{k:s[k] for k in ('sample_id','topic','essay')} for s in samples],destination/'human_reference')
    dump(destination/'study_manifest.json',{
        'dataset_sha256':digest,'created_at':datetime.now(timezone.utc).isoformat(),
        'status':'prepared_not_executed','call_bound':430,'suites':{'reliability':120,'ablation':310},
        'primary_comparison':'all_feedback_vs_no_feedback',
        'reliability_repeats':3,'ablation_repeats':1,'max_attempts':1,'max_rounds':2,
        'failure_policy':'Retain all jobs; no automatic reruns; report failed and missing separately',
        'reference_policy':'One actual human or mean of two independent human raters; preserve original differences',
        'quality_policy':'Weighted human-score difference per sample; positive/tie/negative at tolerance 1e-12; no practical significance implied',
        'ci_policy':'Exploratory sample-level percentile bootstrap; repeated calls are not independent samples',
        'scope':'Ten AI-authored synthetic texts; all dev; no held-out or complete-system validation',
        'budget_note':'Request-attempt cap only; not monetary cap; no price assumed'})
    return destination


def progress(study):
    result={'study':str(study),'suites':{}}
    for name in ('reliability','ablation'):
        directory=Path(study)/name
        if (directory/'plan.json').exists():
            plan,jobs=load_jobs(directory)
            result['suites'][name]={'planned':len(plan['jobs']),'completed':len(jobs),
                'successful':sum(j['result'].get('final_score') is not None for j in jobs),
                'calls':sum(len(j['result'].get('calls',[])) for j in jobs)}
        else:result['suites'][name]={'status':'not_started'}
    return result


def summarize(study):
    study=Path(study)
    reports={}
    for name in ('reliability','ablation'):
        if (study/name/'plan.json').exists():
            reports[name]=analyze(study/name)
            path=study/(name+'_model_report.json')
            if not path.exists():dump(path,reports[name])
    lines=['# EssayLoop n=10 exploratory study','',
        'AI-authored synthetic texts, all dev, no held-out validation.',
        'Actual model records only. Independent human labels NOT PROVIDED.',
        'Human MAE/Spearman, rater consistency and human quality improvements remain pending.','']
    if 'reliability' in reports:
        r=reports['reliability']['reliability']
        lines += ['## Repeated ratings',f"Completed {r['completed_jobs']}; failed {r['failed_jobs']}",'',
            '|Sample|Successful/planned|Relevance SD|Grammar SD|Structure SD|Depth SD|Total min-max|',
            '|---|---|---|---|---|---|---|']
        def fmt(v):return 'null' if v is None else f'{v:.4f}'
        for row in r['per_sample']:
            sd=[fmt(row['dimensions'][d]['sample_sd']) for d in ('relevance','grammar','structure','depth')]
            total=row['total_score']
            lines.append('|'+'|'.join([row['sample_id'],str(row['successful_repeats'])+'/'+str(row['planned_repeats']),
                *sd,fmt(total['min'])+' - '+fmt(total['max'])])+'|')
    if 'ablation' in reports:
        lines += ['','## Ablation execution: human quality pending','',
            '|Variant|Completed|Successful|Failed|Mean model self-score (descriptive only)|','|---|---|---|---|---|']
        for name,v in reports['ablation']['variants'].items():
            lines.append(f"|{name}|{v['completed']}|{v['successful']}|{v['failed']}|{v['model_mean_score_descriptive_only']}|")
        lines += ['','Primary comparison: all_feedback vs no_feedback.',
            'Do not infer quality gain or improved/tied/regressed human counts without blind labels.',
            'Retain negative results and interpret successful-pair comparisons together with failures.']
    lines += ['','Use src.annotation analyze-reference / analyze-ablation after actual human labeling.',
        'Keep private mapping and model reports hidden from human annotators.',
        'With ten AI-authored examples and one rewrite realization per condition, all findings are exploratory.']
    path=study/'EXPLORATORY_REPORT.md'
    if not path.exists():path.write_text('\n'.join(lines),encoding='utf-8')
    return reports


def run_study(study,*,execute=False,max_calls=None):
    study=Path(study)
    if not execute or type(max_calls) is not int or max_calls!=430:
        raise ValueError('Requires explicit --execute and --max-calls 430')
    if any((study/name).exists() for name in ('reliability','ablation')):
        raise ValueError('Existing run directories; no automatic paid replay')
    for name,bound in (('reliability',120),('ablation',310)):
        plan=json.loads((study/(name+'_plan.json')).read_text(encoding='utf-8'))
        validate_plan(plan)
        if plan['max_call_attempts']!=bound:raise ValueError('Unexpected suite budget')
        execute_plan(plan,study/name,execute=True,max_calls=bound)
    code=annotation_main(['export-ablation','--run-dir',str(study/'ablation'),'--output-dir',str(study/'human_ablation')])
    if code:raise ValueError('Blind packet export failed; original model outputs retained')
    summarize(study)
    dump(study/'study_completion.json',{'status':'model_runs_completed_human_labels_pending',
        'completed_at':datetime.now(timezone.utc).isoformat(),'progress':progress(study)})


def main(argv=None):
    p=argparse.ArgumentParser(description='Ten-text exploratory study; no fabricated human labels')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--model',default='qwen3.8-27b')
    a=sub.add_parser('run');a.add_argument('--study-dir',type=Path,required=True);a.add_argument('--execute',action='store_true');a.add_argument('--max-calls',type=int,required=True)
    for name in ('status','summarize'):
        a=sub.add_parser(name);a.add_argument('--study-dir',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='prepare':print(prepare(args.output_dir,args.model))
    elif args.command=='run':run_study(args.study_dir,execute=args.execute,max_calls=args.max_calls)
    elif args.command=='summarize':summarize(args.study_dir)
    else:print(json.dumps(progress(args.study_dir),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
