#!/usr/bin/env python3
"""Level Three evaluator and agent metrics.

Evaluates turn-level and trajectory-level answers with a local Transformers
judge, and computes tool/latency/output metrics directly from agent traces.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, sys, time
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs): return iterable if iterable is not None else []
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DATA_ROOT = ROOT / "evaluation_data"
DEFAULT_GOLD = EVALUATION_DATA_ROOT / "gold/Level_three_v3_en_agent.jsonl"
DEFAULT_PRED = ROOT / "outputs/level_three/Qwen3-8B/Qwen3-8B.jsonl"
DEFAULT_TEMPLATE = EVALUATION_DATA_ROOT / "prompts/AnesTRACE-Eval-Data.jsonl"
DEFAULT_TURN_SYSTEM_PROMPT = EVALUATION_DATA_ROOT / "prompts/anestrace_eval_turn_system_en.txt"
DEFAULT_TRAJECTORY_SYSTEM_PROMPT = EVALUATION_DATA_ROOT / "prompts/anestrace_eval_trajectory_system_en.txt"
DEFAULT_MODEL = ROOT / "../LlamaFactory/outputs/qwen35_9b_lora_dpo_from_merged_lr3e6_1epoch/merged_checkpoint-105"
PROMPT_VERSION = "anestrace-level3-local-judge.v1"
EVALUATION_LEVELS = ("turn", "trajectory", "both")
TURN_INSTRUCTION = ("【Evaluation Turn-Level】Evaluate exactly intraoperative multi-steps decision-making turn by separately assessing the candidate diagnosis and intervention decision using the current patient state, relevant history, recorded prior action, reference response, and candidate response. Apply the rubric and output schema defined in the system prompt.")
TRAJ_INSTRUCTION = ("【Evaluation Trajectory-Level】Evaluate one complete intraoperative multi-steps decision-making episode for longitudinal consistency across turns, focusing on temporal evidence and response adaptation and longitudinal management coherence. Apply the rubric and output schema defined in the system prompt.")

def load_jsonl(path: Path) -> List[dict]:
    rows=[]
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try: rows.append(json.loads(line))
                except Exception as e: raise ValueError(f"invalid JSON {path}:{n}: {e}")
    return rows

def dump_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")

def sha(value: Any) -> str:
    return hashlib.sha256((value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,sort_keys=True)).encode()).hexdigest()

def text(v: Any) -> str:
    if v is None: return ""
    if isinstance(v,str): return v
    return json.dumps(v,ensure_ascii=False,indent=2)

def load_template(path: Path) -> Tuple[str,str,str,str]:
    part_system=""; turn_system=""; traj_system=""; turn=""; traj=""
    for row in load_jsonl(path):
        ins=str(row.get("instruction", "")); system=str(row.get("system", ""))
        if "Turn-Level" in ins:
            if not turn: turn=ins
            if not turn_system and system: turn_system=system
        if "Trajectory-Level" in ins or "Trajectory" in ins:
            if not traj: traj=ins
            if not traj_system and system: traj_system=system
        if not part_system and system: part_system=system
    return (turn_system or part_system, turn or TURN_INSTRUCTION,
            traj or TRAJ_INSTRUCTION, traj_system or turn_system or part_system)

def input_block(ep:dict, turn:dict) -> str:
    p=ep.get("patient_information",{})
    x=turn.get("input",{}) or {}
    procedure=p.get("procedure_anesthesia_context",p.get("procedure_and_anesthesia_context",{}))
    return "\n".join([
        "[Patient Context]", text(p.get("patient_profile",p)),
        "[Procedure & Anesthesia Context]", text(procedure),
        "[Current Decision Point]", text(turn.get("decision_point",{})),
        "[Anesthesia & Infusions]", text(x.get("anesthesia_medication_state",{})),
        "[Recent Vital-Sign Trends]", text(x.get("vital_sign_trends",{})),
        "[Visible Test Results]", text(x.get("visible_test_results",{})),
        "[Recorded Action]", text(x.get("recorded_action",{})),
        "[Recorded Action Before Current Turn]", text(x.get("previous_intervention",x.get("history",{}))),
    ])

def pred_parts(row:dict) -> Tuple[str,str]:
    p=row.get("prediction",{}) or {}
    return text(p.get("state_assessment",p.get("diagnosis",{}))), text(p.get("expert_recommendation",p.get("intervention_decision",{})))

def ref_parts(turn:dict) -> Tuple[str,str]:
    a=turn.get("answer",{}) or {}
    return text(a.get("state_assessment",{})), text(a.get("acceptable_plans",a.get("natural_language_answer",{})))

def build_turn_prompt(ep:dict, turn:dict, pred:dict, instruction:str) -> str:
    cd,ci=pred_parts(pred); rd,ri=ref_parts(turn)
    return "\n\n".join([input_block(ep,turn), "[Candidate Diagnosis]\n[Clinical Diagnosis]\n"+cd, "[Candidate Intervention Decision]\n[Specific Intervention]\n"+ci, "[Reference Diagnosis]\n[Clinical Diagnosis]\n"+rd, "[Reference Intervention Decision]\n[Specific Intervention]\n"+ri, "[Task]\n"+instruction])

def build_trajectory_prompt(ep:dict, turns:List[dict], pred_by_turn:Dict[str,dict], instruction:str) -> str:
    chunks=["[Patient Context]\n"+text(ep.get("patient_information",{})), "[Procedure & Anesthesia Context]\n"+text(ep.get("procedure_name",{})), "[Trajectory]"]
    for t in turns:
        tid=str(t.get("turn_id","")); cd,ci=pred_parts(pred_by_turn.get(tid,{})); rd,ri=ref_parts(t)
        chunks += [f"[Turn {tid}]", f"[Decision Point]\n{text(t.get('decision_point',{}))}", input_block(ep,t), "[Candidate Diagnosis]\n"+cd, "[Candidate Intervention Decision]\n"+ci, "[Reference Diagnosis]\n"+rd, "[Reference Intervention Decision]\n"+ri]
    chunks.append("[Task]\n"+instruction); return "\n\n".join(chunks)

def parse_json(raw:str)->dict:
    s=(raw or "").strip(); s=re.sub(r"^```(?:json)?\s*|\s*```$","",s,flags=re.I)
    try: return json.loads(s)
    except Exception:
        m=re.search(r"\{.*\}",s,re.S)
        if not m: raise ValueError("judge output is not JSON")
        return json.loads(m.group(0))

def _score(value, label):
    if isinstance(value,bool): raise ValueError(f"invalid {label}")
    try:
        if isinstance(value,str): value=float(value.strip())
        if isinstance(value,float) and not value.is_integer(): raise ValueError
        value=int(value)
    except Exception:
        raise ValueError(f"invalid {label}")
    if value not in (0,1,2): raise ValueError(f"invalid {label}")
    return value

def validate_scores(obj:dict, kind:str)->dict:
    if not isinstance(obj,dict): raise ValueError("judge JSON must be object")
    if kind=='trajectory':
        required=['temporal_evidence_and_response_adaptation','longitudinal_management_coherence']
        out={}
        for part in required:
            d=obj.get(part)
            if not isinstance(d,dict): raise ValueError(f"missing {part}")
            score=d.get('score',d.get('value'))
            out[part]={'score':_score(score,part+'.score'),'rationale':str(d.get('rationale',d.get('reason','')))}
        return out
    required=['diagnosis','intervention_decision']; out={}
    aliases={'d1_clinical_correctness':['d1_clinical_correctness','clinical_correctness','d1'],'d2_evidence_based_reasoning':['d2_evidence_based_reasoning','evidence_based_reasoning','d2'],'d3_task_completeness':['d3_task_completeness','task_completeness','d3']}
    for part in required:
        d=obj.get(part)
        if not isinstance(d,dict): raise ValueError(f"missing {part}")
        vals={}
        for k, names in aliases.items():
            key=next((name for name in names if name in d),None); value=d.get(key) if key else None; rationale=d.get(k+'_rationale',d.get('rationale',''))
            if isinstance(value,dict): rationale=value.get('rationale',value.get('reason',rationale)); value=value.get('score',value.get('value'))
            vals[k]={'score':_score(value,f'{part}.{k}'),'rationale':str(rationale)}
        vals['total_score']=sum(vals[k]['score'] for k in aliases)
        if 'safety' in d: vals['safety']=d['safety']
        out[part]=vals
    if 'safety' in obj:
        out['safety']=obj['safety']
    return out

class LocalJudge:
    def __init__(self, model_path:str, dtype:str='bfloat16', device_map:str='auto', max_new_tokens:int=1024, thinking:bool=False):
        import torch, transformers
        self.torch=torch; self.max_new_tokens=max_new_tokens; self.thinking=thinking
        dt={'bfloat16':torch.bfloat16,'float16':torch.float16,'float32':torch.float32}[dtype]
        cls=getattr(transformers,'Qwen3_5ForConditionalGeneration',transformers.AutoModelForCausalLM)
        self.processor=transformers.AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
        self.model=cls.from_pretrained(model_path,torch_dtype=dt,device_map=device_map,trust_remote_code=True)
        self.model.eval()
    def generate_batch(self, system:str, users:List[str])->List[str]:
        msgs=[[{"role":"system","content":system},{"role":"user","content":u}] for u in users]
        batch=self.processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True,padding=True,enable_thinking=self.thinking)
        device=next(self.model.parameters()).device
        batch={k:(v.to(device) if hasattr(v,"to") else v) for k,v in batch.items()}
        prompt_len=batch["input_ids"].shape[1]
        with self.torch.inference_mode(): out=self.model.generate(**batch,max_new_tokens=self.max_new_tokens,do_sample=False,return_dict_in_generate=True)
        return self.processor.batch_decode(out.sequences[:,prompt_len:],skip_special_tokens=True)
    def generate(self, system:str, user:str)->str:
        msgs=[{"role":"system","content":system},{"role":"user","content":user}]
        p=self.processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True,enable_thinking=self.thinking)
        device=next(self.model.parameters()).device
        p={k:(v.to(device) if hasattr(v,'to') else v) for k,v in p.items()}
        with self.torch.inference_mode(): out=self.model.generate(**p,max_new_tokens=self.max_new_tokens,do_sample=False,return_dict_in_generate=True)
        ids=out.sequences[:,p['input_ids'].shape[1]:]
        return self.processor.batch_decode(ids,skip_special_tokens=True)[0]

def index_data(gold_rows,pred_rows):
    gold={}; pred={}; errors=[]
    for e in gold_rows:
        eid=str(e.get('episode_id',''))
        if eid in gold: errors.append(f'duplicate episode {eid}')
        gold[eid]=e
    for r in pred_rows:
        key=(str(r.get('episode_id','')),str(r.get('turn_id','')))
        if key in pred: errors.append(f'duplicate prediction {key}')
        pred[key]=r
    return gold,pred,errors

def agent_metrics(ep, pred_by_tid):
    rows=[]; required={'get_patient_profile','get_procedure_anesthesia_context','get_anesthesia_medication_state','get_previous_intervention','get_visible_test_results'}
    for t in ep.get('turns',[]):
        tid=str(t.get('turn_id','')); r=pred_by_tid.get(tid,{})
        calls=r.get('tool_calls',[]) or []; models=r.get('model_calls',[]) or []
        names=[str(c.get('tool_name','')) for c in calls]; success=[c for c in calls if str(c.get('status','')).lower() in ('success','ok','completed')]
        context=set(names)&required; knowledge=[n for n in names if n not in required]
        p=r.get('prediction',{}) or {}; out=text(p)
        tool_time=sum(float(c.get('elapsed_seconds') or 0) for c in calls); model_time=sum(float(c.get('elapsed_seconds') or 0) for c in models)
        rows.append({'episode_id':ep.get('episode_id'),'turn_id':tid,'status':r.get('status'),'tool_call_count':len(calls),'successful_tool_call_count':len(success),'tool_success_rate':len(success)/len(calls) if calls else 0.0,'context_tool_coverage':len(context)/len(required),'knowledge_tool_call_count':len(knowledge),'unique_tool_count':len(set(names)),'redundant_tool_call_count':len(names)-len(set(names)),'output_chars':len(out),'output_words':len(out.split()),'model_time_seconds':model_time,'tool_time_seconds':tool_time,'total_observed_time_seconds':model_time+tool_time,'input_tokens':sum(int(c.get('input_tokens') or 0) for c in models),'output_tokens':sum(int(c.get('output_tokens') or 0) for c in models),'total_tokens':sum(int(c.get('total_tokens') or 0) for c in models)})
    return rows

def mean(xs): return sum(xs)/len(xs) if xs else 0.0

def summarize(records, metric_rows, level=None):
    ok=[r for r in records if r.get('status')=='ok']
    summary={'case_count':len(records),'success_count':len(ok),'error_count':len(records)-len(ok)}
    if level=='trajectory':
        vals=[part['score']/2 for r in ok for part in r.get('parts',{}).values() if isinstance(part,dict) and 'score' in part]
        summary.update(scored_count=len(vals),mean_normalized=mean(vals))
    elif level=='turn':
        vals=[part[d]['score']/2 for r in ok for part in r.get('parts',{}).values() for d in ('d1_clinical_correctness','d2_evidence_based_reasoning','d3_task_completeness') if d in part]
        summary.update(scored_count=len(vals),mean_normalized=mean(vals))
    if metric_rows:
        summary['agent_metrics']={k:mean([float(x.get(k,0)) for x in metric_rows]) for k in ('tool_success_rate','context_tool_coverage','output_chars','output_words','model_time_seconds','tool_time_seconds','total_observed_time_seconds','input_tokens','output_tokens','total_tokens')}
        summary['agent_metrics']['row_count']=len(metric_rows)
    return summary

def discover_predictions(root:Path)->List[Path]:
    skip=('.turn_local_judge','.trajectory_local_judge','.agent_metrics','.judge_summary','.agent_summary')
    return sorted(p for p in root.glob('*/*.jsonl') if p.parent.name!='evidence' and not any(x in p.name for x in skip))

def _status_counts(records):
    return dict(sorted(__import__('collections').Counter(str(r.get('status','unknown')) for r in records).items()))

def _dimension_summary(records, part_names, dimensions):
    ok=[r for r in records if r.get('status')=='ok']
    out={'scored_case_count':len(ok)}; scores={d:[] for d in dimensions}; totals=[]
    for r in ok:
        case_vals=[]
        for name in part_names:
            part=(r.get('parts') or {}).get(name,{})
            if not isinstance(part,dict): continue
            for d in dimensions:
                if d in part and isinstance(part[d],dict) and isinstance(part[d].get('score'),int):
                    scores[d].append(part[d]['score']); case_vals.append(part[d]['score'])
            if isinstance(part.get('score'),int): case_vals.append(part['score'])
            if isinstance(part.get('total_score'),int): totals.append(part['total_score'])
        if case_vals and not totals: totals.append(sum(case_vals))
    for d,vals in scores.items(): out[d+'_mean_normalized']=mean(vals)/2 if vals else 0.0
    out['total_mean_normalized']=mean(totals)/(2*len(dimensions)) if totals else 0.0
    if totals: out['total_score_distribution']={str(k):v for k,v in sorted(__import__('collections').Counter(totals).items())}
    return out

def _trajectory_summary(records):
    names=['temporal_evidence_and_response_adaptation','longitudinal_management_coherence']
    ok=[r for r in records if r.get('status')=='ok']; out={'scored_case_count':len(ok)}; scores={n:[] for n in names}; totals=[]
    for r in ok:
        vals=[]
        for n in names:
            part=(r.get('parts') or {}).get(n,{})
            if isinstance(part,dict) and isinstance(part.get('score'),int):
                scores[n].append(part['score']); vals.append(part['score'])
        if len(vals)==len(names): totals.append(sum(vals))
    for n,vals in scores.items(): out[n+'_score_mean_normalized']=mean(vals)/2 if vals else 0.0
    out['total_mean_normalized']=mean(totals)/(2*len(names)) if totals else 0.0
    out['total_score_distribution']={str(k):v for k,v in sorted(__import__('collections').Counter(totals).items())}
    return out

def _turn_safety(records):
    dist=__import__('collections').Counter(); valid=0
    for r in records:
        if r.get('status')!='ok': continue
        parts=r.get('parts') or {}
        safety=parts.get('safety') or (parts.get('intervention_decision') or {}).get('safety')
        severity=safety.get('severity') if isinstance(safety,dict) else safety if isinstance(safety,str) else None
        severity=str(severity).strip().lower() if severity is not None else None
        if severity in ('safe','minor','major','critical'): dist[severity]+=1; valid+=1
    major=dist['major']+dist['critical']
    return {'valid_case_count':valid,'safety_distribution':dict(sorted(dist.items())),'major_or_critical_count':major,'major_or_critical_percentage':(major/valid*100 if valid else 0.0)}

def build_judge_summary(args, pred_path, selected, turns, trajectories, errs):
    evaluation_level=getattr(args,'evaluation_level','both')
    turn_dims=['d1_clinical_correctness','d2_evidence_based_reasoning','d3_task_completeness']
    turn_parts={name:_dimension_summary(turns,[name],turn_dims) for name in ('diagnosis','intervention_decision')}
    trajectory_parts=_trajectory_summary(trajectories)
    all_turn=[part[d]['score'] for r in turns if r.get('status')=='ok' for part in (r.get('parts') or {}).values() for d in turn_dims if isinstance(part.get(d),dict) and isinstance(part[d].get('score'),int)]
    safety=_turn_safety(turns)
    return {'evaluation_level':evaluation_level,'turn_evaluated':evaluation_level in ('turn','both'),'trajectory_evaluated':evaluation_level in ('trajectory','both'),'prompt_version':PROMPT_VERSION,'judge_model':str(args.model),'prediction_file':str(pred_path),'selected_episode_count':len(selected),'turn_record_count':len(turns),'trajectory_record_count':len(trajectories),'turn_status_counts':_status_counts(turns),'trajectory_status_counts':_status_counts(trajectories),'normalization':{'raw_score_min':0,'raw_score_max':2,'normalized_formula':'score / 2'},'turn':{'parts':turn_parts,'overall':{'scored_dimension_count':len(all_turn),'mean_normalized':mean(all_turn)/2 if all_turn else 0.0}},'trajectory':{'dimensions':trajectory_parts},'safety':{'turn':safety,'trajectory':{'valid_case_count':0,'major_or_critical_count':0,'major_or_critical_percentage':0.0}},'alignment_errors':errs}

def build_agent_summary(selected, metrics, pred_index):
    episode_status=[]
    for ep in selected:
        eid=str(ep.get('episode_id','')); rows=[r for (e,_),r in pred_index.items() if e==eid]
        good=sum(str(r.get('status','')).lower() in ('success','success_after_repair','ok') for r in rows)
        episode_status.append(good==len(ep.get('turns',[])) and len(rows)==len(ep.get('turns',[])))
    successful_turns=sum(str(r.get('status','')).lower() in ('success','success_after_repair','ok') for r in metrics)
    names=('tool_success_rate','context_tool_coverage','output_chars','output_words','model_time_seconds','tool_time_seconds','total_observed_time_seconds','input_tokens','output_tokens','total_tokens')
    return {'case_count':len(selected),'success_count':sum(episode_status),'error_count':len(selected)-sum(episode_status),'turn_count':len(metrics),'turn_success_count':successful_turns,'turn_error_count':len(metrics)-successful_turns,'metrics':{k:mean([float(r.get(k,0)) for r in metrics]) for k in names}}

def run_one(args, pred_path:Path, templates:Tuple[str,str,str], judge:Optional[LocalJudge]):
    gold_rows=load_jsonl(Path(args.gold)); pred_rows=load_jsonl(pred_path); gold,pidx,errs=index_data(gold_rows,pred_rows)
    selected=[e for e in gold_rows if not args.split or e.get('split')==args.split][:args.limit or None]
    turn_system,turn_ins,traj_ins,traj_system=templates; evaluation_level=getattr(args,'evaluation_level','both'); evaluate_turn=evaluation_level in ('turn','both'); evaluate_trajectory=evaluation_level in ('trajectory','both'); model_hash=sha(str(args.model)); output_dir=pred_path.parent; stem=pred_path.stem
    turn_out=output_dir/f'{stem}.turn_local_judge.jsonl'; traj_out=output_dir/f'{stem}.trajectory_local_judge.jsonl'; agent_out=output_dir/f'{stem}.agent_metrics.jsonl'; summary_out=output_dir/f'{stem}.judge_summary.json'; agent_summary=output_dir/f'{stem}.agent_summary.json'
    turns=[]; trajectories=[]; metrics=[]; turn_jobs=[]; trajectory_jobs=[]
    for ep in tqdm(selected, desc='prepare episodes', disable=args.no_progress):
        eid=str(ep.get('episode_id','')); pred_by={str(r.get('turn_id','')):r for (e,t),r in pidx.items() if e==eid}
        metrics.extend(agent_metrics(ep,pred_by))
        for t in (ep.get('turns',[]) if evaluate_turn else []):
            tid=str(t.get('turn_id','')); pr=pred_by.get(tid); key=f'{eid}:{tid}'; rec={'episode_id':eid,'turn_id':tid,'split':ep.get('split'),'prompt_version':PROMPT_VERSION,'judge_model':str(args.model),'key':key}
            if pr is None: rec.update(status='error',error='missing prediction'); turns.append(rec); continue
            user=build_turn_prompt(ep,t,pr,turn_ins); rec['prompt_sha256']=sha(user)
            if args.dry_run or judge is None: rec.update(status='dry_run' if args.dry_run else 'pending'); turns.append(rec); continue
            turn_jobs.append((user,rec))
        if not evaluate_trajectory: continue
        if args.dry_run or judge is None: trajectories.append({'episode_id':eid,'status':'dry_run' if args.dry_run else 'pending'}); continue
        user=build_trajectory_prompt(ep,ep.get('turns',[]),pred_by,traj_ins); rec={'episode_id':eid,'split':ep.get('split'),'prompt_sha256':sha(user),'prompt_version':PROMPT_VERSION,'judge_model':str(args.model),'key':eid}
        trajectory_jobs.append((user,rec))
    if judge is not None and not args.dry_run:
        size=max(1,args.batch_size)
        for jobs,kind,target,desc in ((turn_jobs,'turn',turns,'turn judge'),(trajectory_jobs,'trajectory',trajectories,'trajectory judge')):
            for start in tqdm(range(0,len(jobs),size), desc=desc, disable=args.no_progress):
                chunk=jobs[start:start+size]
                try: raw_outputs=judge.generate_batch(turn_system if kind=='turn' else traj_system,[u for u,_ in chunk])
                except Exception as exc: raw_outputs=[None]*len(chunk)
                for (_,rec),raw in zip(chunk,raw_outputs):
                    try:
                        if raw is None: raise ValueError('batch generation failed')
                        rec.update(status='ok',parts=validate_scores(parse_json(raw),kind))
                    except Exception as exc:
                        rec.update(status='error',error=str(exc))
                        if raw is not None: rec['raw_judge_output']=str(raw)
                    target.append(rec)
    if not args.dry_run and not args.agent_metrics_only:
        if evaluate_turn: dump_jsonl(turn_out,turns)
        if evaluate_trajectory: dump_jsonl(traj_out,trajectories)
        summary=build_judge_summary(args,pred_path,selected,turns,trajectories,errs)
        summary_out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.dry_run:
        dump_jsonl(agent_out,metrics)
        agent_summary.write_text(json.dumps({'model':str(args.model),'prediction_file':str(pred_path),'summary':build_agent_summary(selected,metrics,pidx)},ensure_ascii=False,indent=2),encoding='utf-8')
    return len(selected)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default=str(DEFAULT_MODEL)); ap.add_argument('--gold',default=str(DEFAULT_GOLD)); ap.add_argument('--predictions',default=str(DEFAULT_PRED)); ap.add_argument('--template',default=str(DEFAULT_TEMPLATE)); ap.add_argument('--turn-system-prompt',default=str(DEFAULT_TURN_SYSTEM_PROMPT)); ap.add_argument('--trajectory-system-prompt',default=str(DEFAULT_TRAJECTORY_SYSTEM_PROMPT)); ap.add_argument('--split'); ap.add_argument('--limit',type=int); ap.add_argument('--batch-size',type=int,default=1); ap.add_argument('--device'); ap.add_argument('--device-map',default='auto'); ap.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16'); ap.add_argument('--max-new-tokens',type=int,default=1024); ap.add_argument('--enable-thinking',action='store_true'); ap.add_argument('--disable-thinking',action='store_true'); ap.add_argument('--resume',action='store_true'); ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--agent-metrics-only',action='store_true'); ap.add_argument('--evaluation-level',choices=EVALUATION_LEVELS,default='both',help='Judge scope: turn, trajectory, or both (default: both).'); ap.add_argument('--no-progress',action='store_true'); args=ap.parse_args()
    templates=list(load_template(Path(args.template)))
    # Keep the SFT template only for the exact task instructions; system
    # prompts are loaded from the local evaluation_data bundle.
    templates[0]=Path(args.turn_system_prompt).read_text(encoding='utf-8').strip()
    templates[3]=Path(args.trajectory_system_prompt).read_text(encoding='utf-8').strip()
    templates=tuple(templates)
    paths=discover_predictions(ROOT/'outputs/level_three') if args.predictions=='all' else [Path(args.predictions)]
    if not paths: raise SystemExit('no prediction files found')
    judge=None if args.dry_run or args.agent_metrics_only else LocalJudge(args.model,args.dtype,args.device_map,args.max_new_tokens,args.enable_thinking)
    for p in tqdm(paths, desc='models', disable=args.no_progress):
        print(f'evaluating {p}')
        run_one(args,p,templates,judge)

if __name__=='__main__': main()
