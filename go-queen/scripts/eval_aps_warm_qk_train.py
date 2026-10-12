#!/usr/bin/env python3
"""One frozen final warm-QK training-item evaluation. Use its exclusive launcher.

All actually consumed training IDs, each once; six restoration references;
no training, attention observer, retry, generated text or automatic next stage.
"""
from __future__ import annotations
import time
PROGRAM_STARTED = time.perf_counter()
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from train_grounding_microfit import file_sha256,content_sha256,require,relative
from goqueen.train_eval_selection import (observed_slots,build_eval_manifest,validate_eval_manifest,
    questions_from_manifest,restoration_subset,describe,exposure_weighted_summary)
from goqueen.transfer_metrics import summarize_transfer,compare_restoration,align_saved_reference

STAGE='aps_warm_qk_train_eval'
OUTPUT=ROOT/'outputs/aps-warm-qk-train-eval'
MANIFEST=ROOT/'outputs/grounding-data/manifest.json'
REFERENCE=ROOT/'outputs/aps-warm-qk/report.json'
FIT_REPORT=ROOT/'outputs/grounding-assisted-point/report.json'
CHECKPOINT=ROOT/'outputs/aps-warm-qk/adapter.safetensors'
SOURCE_COMPLETION=ROOT/'outputs/aps-warm-qk/process-completion.json'
PINS={
 'outputs/aps-warm-qk/process-completion.json':'2eeb001e3764e6ed83a7d3bc24c2fec22c604687ef59280abb7019ae37113861',
 'outputs/grounding-data/manifest.json':'2c3deae2c0bde1cdd52b105889f0558e295662eb0dc222ffdbf2acbe07bd6279',
 'outputs/aps-warm-qk/report.json':'68456efe52869550ffcf6542c3150c9c739365b43b14ad5ab6bd801fab1b46d2',
 'outputs/grounding-assisted-point/report.json':'2eb4bab5e4e86ff33e586bae94ee010a4cb005a84c0d76ada40903e94d9e75a4',
 'outputs/aps-warm-qk/adapter.safetensors':'0db384223a90fd05f216cf497355d5036ddd667426b9376f2087614b46760e30'}
TOTAL_SECONDS,RESERVE_SECONDS,TOLERANCE=1500,60,1e-5
MAX_FORWARDS=2708

class StopEvaluation(TimeoutError):pass

class ForwardCounter:
    """Transparent model-call wrapper; does not inspect or modify attention or outputs."""
    def __init__(self,model,report):self.model,self.report=model,report
    def __getattr__(self,name):return getattr(self.model,name)
    def __call__(self,*args,**kwargs):
        require(self.report['forward_attempts']<MAX_FORWARDS,'Forward ceiling reached')
        self.report['forward_attempts']+=1
        output=self.model(*args,**kwargs)
        self.report['forward_returns']+=1
        return output

def restore_allows_training(comparison):
    if not isinstance(comparison,dict):return False
    delta=comparison.get('max_abs_logp_difference')
    return (comparison.get('passed') is True and comparison.get('coverage_complete') is True
        and comparison.get('predictions_match') is True and comparison.get('tolerance')==TOLERANCE
        and all(type(comparison.get(k)) is int and comparison[k]==6 for k in ('current_questions','saved_questions'))
        and type(delta) in (int,float) and math.isfinite(delta) and 0<=delta<=TOLERANCE)

def prepare_selection(data,reference,pins):
    require(reference['status']=='complete' and reference['run_completed'] is True,'Not a valid final source run')
    slots=observed_slots(reference['step_attempts'])
    require(slots==reference['train_order']==data['train_order'],'Actual slots and schedules differ')
    val=reference['validation_questions']
    require(len(val)==54,'Saved comparison set changed')
    final=[e for e in reference['evaluations'] if e['name']=='final']
    require(len(final)==1 and final[0]['updates_completed']==1000 and final[0]['coverage_complete']
            and len(final[0]['rows'])==54,'Incomplete final reference')
    manifest=build_eval_manifest(slots,data['train_order'],data['training_pool_questions'],val,
        order_sha256=reference['train_order_sha256'],source=pins)
    validate_eval_manifest(manifest)
    require(manifest['unique_questions']==2702 and manifest['slots']==4000,'Seen question counts changed')
    counts=manifest['counts']
    require(counts['by_class']=={'empty':{'questions':1334,'slots':1334},
        'black':{'questions':690,'slots':1333},'white':{'questions':678,'slots':1333}},'Class counts changed')
    require(counts['by_class_and_exposure']=={'empty:1':1334,'black:1':47,'black:2':643,
        'white:1':23,'white:2':655},'Exposure counts changed')
    require((counts['games'],counts['positions'],counts['point_indices'])==(5,30,360),'Coverage changed')
    require(manifest['separation']['comparison_prompts_in_list']==54
        and len({q['prompt'] for q in val})==52,'Prompt row/string denominator changed')
    saved=describe(align_saved_reference(val,final[0]['rows']),val)
    for k in ('forced_choice_correct','confusion_matrix','macro_recall','three_label_ce','full_vocabulary_ce'):
        require(saved['summary'][k]==final[0][k],'Saved metric differs: '+k)
    restored=restoration_subset(final[0]['rows'])
    lookup={q['question_id']:q for q in val}
    require(len(restored)==6,'Restoration count changed')
    return manifest,saved,restored,[lookup[r['question_id']] for r in restored]

def load_frozen_final_model(root, fit, reference, check_time):
    """Load the saved natural model directly; capture all101 before freezing."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from goqueen.fusion import GoQueen, GatedBoardAttention
    from goqueen.checkpoints import load_adapter_checkpoint
    from goqueen.warm_qk import adapter_parameters
    from train_grounding_microfit import file_sha256, require, parameter_digest, LAYERS

    check_time()
    local = root / fit["local_model"]["directory"]
    require(file_sha256(local / "config.json") == fit["local_model"]["config_sha256"],
            "Pinned language-model configuration changed")
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
    language = AutoModelForCausalLM.from_pretrained(
        local, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa")
    language.gradient_checkpointing_disable()
    torch.manual_seed(42)
    model = GoQueen(language, LAYERS, 768, 8)
    named = adapter_parameters(model)
    require(len(named) == 101 and sum(p.numel() for _, p in named) == 40438795,
            "Full adapter layout changed")
    checkpoint = root / "outputs/aps-warm-qk/adapter.safetensors"
    require(file_sha256(checkpoint) == reference["checkpoint"]["sha256"],
            "Pinned final checkpoint changed")
    metadata = load_adapter_checkpoint(model, checkpoint)
    require(metadata == reference["checkpoint_metadata"]
            and metadata.get("variant") == "m"
            and metadata.get("training_recipe") == "aps_warm_qk"
            and metadata.get("engineering_only") == "false"
            and metadata.get("updates") == "1000", "Wrong final checkpoint metadata")
    model.requires_grad_(False)
    model.eval()
    require(all(p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad
                and p.grad is None for p in model.parameters()), "Require frozen CPU FP32")
    require(all(bool(torch.isfinite(p).all()) for _, p in named), "Non-finite adapter")
    require(all(type(layer.adapter) is GatedBoardAttention for layer in model.injected_layers),
            "Require the original natural attention path")
    base_sha = parameter_digest(model.base_parameters)
    require(base_sha == reference["base_sha256_after"], "Base differs from saved final state")
    selected = set(reference["trainable_names"])
    require(len(selected) == 44 and selected <= {n for n, _ in named}, "Stored subset changed")
    frozen = [(n,p) for n,p in named if n not in selected]
    require(len(frozen) == 57 and parameter_digest(p for _,p in frozen)
            == reference["frozen_adapter_sha256_after"], "Frozen adapter subset differs")
    require(all(parameter_digest([p]) == reference["trainable_sha256_after"][n]
                for n,p in named if n in selected), "Saved final Q/K bytes differ")
    return model, tokenizer, named, {
        "checkpoint_metadata": metadata,
        "base_sha256_before": base_sha,
        "adapter_sha256_before": parameter_digest(p for _,p in named),
        "final_parameter_digests_match": True,
        "local_model_config_sha256": {relative(local / "config.json"): file_sha256(local / "config.json")},
    }


def load_question_features(root, questions, data, reference, check_time):
    """Use existing expert caches; no new extraction, point selector or hook."""
    import numpy as np
    import torch
    from goqueen.grounding_data import point_to_gtp, occupancy_prompt, _absolute_labels
    from train_grounding_microfit import file_sha256, require, relative, parameter_digest

    positions = {p["position_id"]:p for p in data["positions"]}
    grouped = {}
    for q in questions:
        require(q["point"] == point_to_gtp(q["point_index"])
                and q["prompt"] == occupancy_prompt(q["to_play"],q["point"]),
                "Question coordinate or prompt changed")
        grouped.setdefault(q["position_id"], []).append(q)
    features, pins = {}, {}
    for pid, group in grouped.items():
        check_time()
        position = positions[pid]
        path = root / position["source_feature_file"]
        relative_path = relative(path)
        digest = file_sha256(path)
        require(digest == position["feature_sha256"] == reference["feature_sha256"][relative_path],
                "Pinned expert cache changed")
        require(_absolute_labels(path,position["to_play"]) == position["absolute_board_labels"],
                "Board facts disagree with saved inputs")
        for q in group:
            require(q["game_id"] == position["game_id"]
                    and q["to_play"] == position["to_play"]
                    and q["label"] == position["absolute_board_labels"][q["point_index"]],
                    "Question facts disagree with the saved board")
        with np.load(path,allow_pickle=False) as archive:
            arrays = [archive[f"trunk_after_block{i}"] for i in range(11)]
            require(all(a.shape == (361,768) and a.dtype == np.float32
                        and np.isfinite(a).all() for a in arrays), "Invalid feature cache")
            features[pid] = torch.from_numpy(np.stack(arrays))
        require(not features[pid].requires_grad, "Features must be detached")
        pins[relative_path] = digest
    return features, pins, {pid:parameter_digest([f]) for pid,f in features.items()}


def final_integrity(model,named,features,before):
    from train_grounding_microfit import parameter_digest
    from train_aps_warm_qk import binding_clear
    base = parameter_digest(model.base_parameters)
    adapters = parameter_digest(p for _,p in named)
    feature_sha = {pid:parameter_digest([f]) for pid,f in features.items()}
    return {
        "base_sha256_after":base,
        "adapter_sha256_after":adapters,
        "feature_memory_sha256_after":feature_sha,
        "base_unchanged":base == before["base_sha256_before"],
        "adapters_unchanged":adapters == before["adapter_sha256_before"],
        "features_unchanged":feature_sha == before["feature_memory_sha256_before"],
        "all_parameters_frozen_and_gradients_absent":all(
            not p.requires_grad and p.grad is None for p in model.parameters()),
        "runtime_binding_cleared":binding_clear(model),
    }

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=TOTAL_SECONDS)
    parser.add_argument('--origin-start',type=float,required=True)
    args=parser.parse_args(argv)
    require(args.seconds==TOTAL_SECONDS and math.isfinite(args.origin_start)
        and 0<args.origin_start<=PROGRAM_STARTED,'Use fixed total budget and valid origin')
    origin,deadline=args.origin_start,args.origin_start+args.seconds
    parent=json.loads((OUTPUT/'process-start.json').read_text())
    source_sha=file_sha256(__file__)
    require(parent['launcher_pid']==os.getppid() and parent['origin_monotonic']==origin
        and parent['seconds']==args.seconds and parent['runner_sha256']==source_sha,'Use exclusive launcher')
    path,temp=OUTPUT/'report.json',OUTPUT/'report.tmp'
    require(not any(p.exists() for p in [path,temp,OUTPUT/'run-start.json']), 'Existing evaluation retained')
    with (OUTPUT/'run-start.json').open('x') as f:
        json.dump({'stage':STAGE,'seconds':args.seconds,'runner_sha256':source_sha,'automatic_retry':False},f)
    with (OUTPUT/'runner-at-run.py').open('xb') as f:f.write(Path(__file__).read_bytes())
    report={'stage':STAGE,'status':'running','valid_complete':False,'error':None,
        'source_script_sha256':source_sha,'automatic_retry':False,
        'runtime':{'device':'cpu','dtype':'float32','threads':4,'batch_size':1,'sdpa':'math',
            'soft_total_seconds':args.seconds,'closing_reserve_seconds':RESERVE_SECONDS},
        'scope':'Fixed finite seen-training list at final weights. Comparison with saved54 is descriptive, not matched, blind or causal. No baseline improvement on this training list is measured.',
        'partial_scope':'Completed hash-order prefix only; not an unbiased sample or full-list estimate.',
        'optimizer_constructed':False,'optimizer_updates':0,'backward_calls':0,
        'forward_attempts':0,'forward_returns':0,'training_rows':[],'restoration_rows':[],
        'restoration':None,'summary':None,'exposure_weighted':None,'training_started':False}
    model,named,features,before=None,None,None,None
    data,reference,manifest,pins=None,None,None,None
    phase,slowest='source_validation',0.
    previous=None

    def record():
        report['seconds_before_report_write']=time.perf_counter()-origin
        report['budget_exceeded']=report['seconds_before_report_write']>args.seconds
        if report['budget_exceeded']:report['valid_complete']=False
        temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
        after=time.perf_counter()-origin
        if after>args.seconds and not report['budget_exceeded']:
            report.update(valid_complete=False,budget_exceeded=True,seconds_after_initial_report_write=after)
            temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)

    def expired(_signum,_frame):raise StopEvaluation('soft_deadline')
    def time_available():
        if time.perf_counter()+RESERVE_SECONDS+max(slowest,1.)>=deadline:
            raise StopEvaluation('insufficient_remaining_time_with_closing_reserve')
    def arm_closing():signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.perf_counter()))

    try:
        previous=signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-RESERVE_SECONDS-time.perf_counter()))
        record()
        pins={rel:file_sha256(ROOT/rel) for rel in PINS}
        require(pins==PINS,'Pinned source input changed')
        data,reference,fit=[json.loads(p.read_text()) for p in (MANIFEST,REFERENCE,FIT_REPORT)]
        source_completion=json.loads(SOURCE_COMPLETION.read_text())
        require(source_completion['valid_complete'] is True and source_completion['exit_code']==0
            and source_completion['report_sha256']==pins['outputs/aps-warm-qk/report.json'],
            'Original final run external completion differs')
        manifest,saved54,saved_restore,restore_questions=prepare_selection(data,reference,pins)
        questions=questions_from_manifest(manifest)
        report.update(training_questions=questions,restoration_questions=restore_questions,
            input_sha256=pins,eval_manifest_content_sha256=manifest['content_sha256'],
            saved54=saved54,prompt_overlap={'comparison_rows':54,'distinct_strings':52,'distinct_prompt_intersection':len({q['prompt'] for q in questions}&{q['prompt'] for q in reference['validation_questions']})},
            references={'uniform_three_label_ce':math.log(3),'unique_empty_share':1334/2702,
                'exposure_empty_share':1334/4000,'no_statistical_resolution_or_outcome_bands':True})
        with (OUTPUT/'eval-manifest.json').open('x') as f:
            json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        report['eval_manifest_file_sha256']=file_sha256(OUTPUT/'eval-manifest.json')
        phase='imports'
        import torch
        import transformers
        from goqueen import fusion,warm_qk,checkpoints,grounding_data
        from goqueen.grounding_eval import score_questions
        from train_grounding_assisted_point import imported_first_party
        from train_aps_warm_qk import binding_clear
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        report['versions']={'torch':torch.__version__,'transformers':transformers.__version__}
        source_hashes=imported_first_party()
        for rel in ['src/goqueen/fusion.py','src/goqueen/grounding_eval.py','src/goqueen/smoke_data.py',
                    'src/goqueen/checkpoints.py','src/goqueen/warm_qk.py','src/goqueen/grounding_data.py']:
            require(source_hashes[rel]==reference['implementation_sha256'][rel], 'Runtime source changed: '+rel)
        report['implementation_sha256']=source_hashes
        for rel,digest in source_hashes.items():
            target=OUTPUT/'source-at-run'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write((ROOT/rel).read_bytes())
            require(file_sha256(target)==digest,'Source copy mismatch')
        record();time_available()
        phase='load_features'
        features,feature_pins,feature_memory=load_question_features(ROOT,questions+restore_questions,data,reference,time_available)
        report['feature_sha256']=feature_pins
        phase='load_final_natural_model'
        model,tokenizer,named,before=load_frozen_final_model(ROOT,fit,reference,time_available)
        before['feature_memory_sha256_before']=feature_memory
        report.update(before)
        require(binding_clear(model),'Initial bindings not clear')
        counter=ForwardCounter(model,report)
        record()
        def score_one(q,key,stream):
            nonlocal slowest
            time_available();started=time.perf_counter()
            rows=score_questions(counter,tokenizer,[q],features,batch_size=1,deadline=deadline-RESERVE_SECONDS)
            slowest=max(slowest,time.perf_counter()-started)
            if not rows:raise StopEvaluation('scorer_deadline_before_forward')
            require(len(rows)==1 and binding_clear(model),'Incomplete score or uncleared bindings')
            summarize_transfer(rows,[q])
            # Retain the in-memory row before append: final report can still record it after an I/O error.
            report[key].append(rows[0])
            stream.write(json.dumps(rows[0],ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            report['slowest_scoring_call_seconds']=slowest
        phase='restore_six'
        with (OUTPUT/'restoration-rows.jsonl').open('x') as stream:
            for q in restore_questions:score_one(q,'restoration_rows',stream)
        report['restoration']=compare_restoration(report['restoration_rows'],saved_restore,TOLERANCE)
        require(restore_allows_training(report['restoration']),'Saved final endpoint restoration failed')
        report['training_started']=True;record();phase='score_seen_questions'
        with (OUTPUT/'training-rows.jsonl').open('x') as stream:
            for index,q in enumerate(questions,1):
                score_one(q,'training_rows',stream)
                if index%200==0:
                    record();print(json.dumps({'completed_training_rows':index,'requested':2702}),flush=True)
        report['status']='complete'
    except StopEvaluation as exc:
        report.update(status='stopped',error=str(exc).replace(str(ROOT),'.'),failed_phase=phase)
    except Exception as exc:
        report.update(status='error',error=type(exc).__name__+': '+str(exc).replace(str(ROOT),'.'),failed_phase=phase)
    finally:
        try:
            arm_closing()
            if model is not None and before is not None and features is not None:
                report.update(final_integrity(model,named,features,before))
            if manifest is not None:
                report['summary']=describe(report['training_rows'],questions_from_manifest(manifest))
                report['exposure_weighted']=exposure_weighted_summary(report['training_rows'],manifest)
                selected=questions_from_manifest(manifest)
                completed=selected[:len(report['training_rows'])]
                report['prefix_composition']={field:{'requested':dict(Counter(str(q[field]) for q in selected)),
                    'completed':dict(Counter(str(q[field]) for q in completed))}
                    for field in ('label','game_id','position_id')}
                report['prefix_composition']['exposure_count']={
                    'requested':dict(Counter(str(e['exposure_count']) for e in manifest['entries'])),
                    'completed':dict(Counter(str(e['exposure_count']) for e in manifest['entries'][:len(completed)]))}
                report['exposure_strata_interpretation']='Accounting only; exposure strata have different class composition and recency, not a repeated-training effect.'
                reread_data=json.loads(MANIFEST.read_text());reread_ref=json.loads(REFERENCE.read_text())
                rebuilt,*_=prepare_selection(reread_data,reread_ref,pins)
                report['manifest_rebuilt_equal']=(rebuilt==manifest
                    and json.loads((OUTPUT/'eval-manifest.json').read_text())==manifest
                    and file_sha256(OUTPUT/'eval-manifest.json')==report['eval_manifest_file_sha256'])
            check_pins={**(pins or {}),**report.get('feature_sha256',{}),**report.get('implementation_sha256',{}),**report.get('local_model_config_sha256',{})}
            report['source_hashes_unchanged']=bool(check_pins) and all(file_sha256(ROOT/rel)==digest for rel,digest in check_pins.items())
            conditions={k:report.get(k) is True for k in ('base_unchanged','adapters_unchanged','features_unchanged',
                'all_parameters_frozen_and_gradients_absent','runtime_binding_cleared','source_hashes_unchanged','manifest_rebuilt_equal')}
            conditions.update(restoration=restore_allows_training(report.get('restoration')),
                exact_counts=report['forward_attempts']==report['forward_returns']==MAX_FORWARDS
                    and len(report['training_rows'])==2702 and len(report['restoration_rows'])==6,
                complete_status=report['status']=='complete')
            report['complete_endpoint_conditions']=conditions
            report['valid_complete']=all(conditions.values())
        except Exception as exc:
            report.update(valid_complete=False,closing_error=type(exc).__name__+': '+str(exc).replace(str(ROOT),'.'))
            if report['status']=='complete':report['status']='error'
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            if previous is not None:signal.signal(signal.SIGALRM,previous)
            record()
    print(json.dumps({'stage':STAGE,'status':report['status'],'valid_complete':report['valid_complete'],
        'training_rows':len(report['training_rows']),'forward_attempts':report['forward_attempts'],
        'seconds_before_report_write':report['seconds_before_report_write']},allow_nan=False),flush=True)
    return 0 if report['valid_complete'] else 1

if __name__=='__main__':raise SystemExit(main())
