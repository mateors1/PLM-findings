"""Read-only CPU diagnosis of the frozen v1 parent admission failure."""
from __future__ import annotations
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/learning/symmetric-affine8000-v1'
PARENT=ROOT/'runs/national_dex_continuation_control_s1729_v1'
EXPECTED_PARENT='e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1'
EXPECTED_HISTORY='e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8'
EXPECTED_RUNNER='a9899d4c494198afcf35346903f93bc39d5771d60a6711dc164d12a42ad11568'
inputs={}
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
def bind(path,expected=None):
    digest=sha(path)
    if expected is not None and digest!=expected:
        raise ValueError(f'identity mismatch: {path}')
    inputs[str(path.relative_to(ROOT))]=digest
    return path
def read(path,expected=None):
    return json.loads(bind(path,expected).read_text(encoding='utf-8'))
def write(path,value):
    with path.open('xb') as stream:
        stream.write((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())

bind(Path(__file__))
frozen=read(RUN/'summary.json')
training=read(RUN/'training.json',frozen['artifact_sha256']['training.json'])
bind(ROOT/'scripts/refit_symmetric_affine.py',EXPECTED_RUNNER)
bind(RUN/'summary.script.py',EXPECTED_RUNNER)
historical=read(ROOT/'runs/learning/bilinear-budget8000-v1/summary.json',EXPECTED_HISTORY)
run=read(PARENT/'run.json','ec6c8e358586efa5808591ca0112ea13b4e52cae4b9891ccac82c299332b870d')
sidecar=read(PARENT/'checkpoint-final.pt.json','4a0cbb69e0f0fb9886612f6ca66ca92858cc96060d86551588ddab7eb9f9756d')
checkpoint=bind(PARENT/'checkpoint-final.pt',EXPECTED_PARENT)
# Only archived source files are consulted; no model is constructed or forwarded.
runtime_src=RUN/'runtime/src'
for relative in ('plm/config.py','plm/training/checkpoint.py','plm/training/trainer.py'):
    path=runtime_src/relative
    bind(path,frozen['runtime_files_sha256'][str(path.resolve())])
sys.path.insert(0,str(runtime_src))
from plm.config import ModelConfig, RootConfig
import torch
assert not torch.cuda.is_initialized()
payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert not torch.cuda.is_initialized()
metadata_fields=('config','experiment_identity','corpus_identity','split_hash','training_metadata')
sidecar_checks={name:sidecar[name]==payload[name] for name in metadata_fields}
sidecar_checks['global_step']=sidecar['global_step']==payload['global_step']
assert all(sidecar_checks.values())
assert all(t.device.type=='cpu' for t in payload['model'].values())
raw={k:payload[k] for k in metadata_fields}
identity=run['identity']
root_config=RootConfig.model_validate(run['config'])
model_config=root_config.model.model_copy(update={'vocab_size':2049,'max_seq_len':root_config.train.seq_len})
checks={
 'global_step_2000':payload['global_step']==2000,
 'checkpoint_hash_original':sha(checkpoint)==EXPECTED_PARENT,
 'checkpoint_config_equals_run_root_config':raw['config']==run['config'],
 'corpus_identity_equals_historical':raw['corpus_identity']==historical['corpus_identity'],
 'training_metadata_equals_sidecar':raw['training_metadata']==sidecar['training_metadata'],
 'training_objective_causal_next_token':raw['training_metadata']['objective']=='causal-next-token-v1',
 'training_model_config_equals_resolved_model':ModelConfig.model_validate(raw['training_metadata']['model_config'])==model_config,
 'training_evaluator_current':identity['evaluator_version']=='plm-train-causal-v2',
 'training_identity_corpus_fields_match':all(identity[k]==v for k,v in historical['corpus_identity'].items()),
}
assert [name for name,ok in checks.items() if not ok]==['checkpoint_config_equals_run_root_config']
assert raw['config']==run['config']['train']
module_hashes={}
for name,module in sorted(sys.modules.items()):
    if name=='plm' or name.startswith('plm.'):
        path=Path(module.__file__).resolve()
        assert path.is_relative_to(runtime_src)
        module_hashes[name]=sha(bind(path,frozen['runtime_files_sha256'][str(path)]))
report={
 'schema_version':1,'diagnosis_id':'symmetric-affine8000-failure-diagnosis-v1',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'failed_campaign':'symmetric-affine8000-v1','diagnosis_complete':True,
 'read_only_originals':True,'candidate_retry_executed':False,'neural_forward_executed':False,
 'training_executed':False,'optimizer_steps_executed':0,'gpu_used':False,
 'cuda_initialized_after_cpu_load':torch.cuda.is_initialized(),
 'corpus_graph_or_protected_files_opened':False,
 'cpu_checkpoint_load':{'sha256':EXPECTED_PARENT,'map_location':'cpu','model_state_count':len(payload['model']),
                        'sidecar_payload_metadata_equal':sidecar_checks},
 'admission_predicates':checks,
 'sole_failing_predicate':'raw["config"] == context["parent_run"]["config"]',
 'root_cause':'Checkpoint config stores TrainConfig; run.json config stores RootConfig. The v1 adapter compared different configuration schemas.',
 'verified_repair_equivalence':{'checkpoint_config_equals_run_train_config':True,
     'checkpoint_training_model_config_equals_resolved_root_model_config':True,
     'checkpoint_identity_equals_original_run_identity':raw['experiment_identity']==identity,
     'inherited_architecture_matches_resolved_model':historical['inherited_architecture']==model_config.model_dump(mode='json')},
 'configuration_keys':{'checkpoint_config':sorted(raw['config']),'run_root_config':sorted(run['config']),
                       'run_train_config':sorted(run['config']['train'])},
 'source_evidence':{'frozen_runner':'scripts/refit_symmetric_affine.py:466',
     'archived_trainer':'runs/learning/symmetric-affine8000-v1/runtime/src/plm/training/trainer.py:342',
     'archived_save_argument':'config=self.config (Trainer receives TrainConfig)',
     'archived_loader':'runs/learning/symmetric-affine8000-v1/runtime/src/plm/training/checkpoint.py:163'},
 'failure_stage':{'runtime_returned':False,'outer_model_available':False,'outer_optimizer_available':False,
                  'parent_model_loaded_on_cpu':True,'residual_parameters_attached':False,
                  'completed_updates':training['completed_updates'],'training_history_length':len(training['history'])},
 'recommended_v2_repair':[
   'Preserve frozen v1 artifacts and declare a separately versioned adapter/runner attempt.',
   'Compare authoritative payload config to authenticated parent_run.config.train, not the entire root config.',
   'Keep original checkpoint hash, complete identity, corpus/split, training objective, model configuration, step count and sidecar-payload equality checks unchanged.',
   'Add a synthetic actual-schema fixture: RootConfig in run.json and TrainConfig in checkpoint; require success only for exact matching train config and reject tampered optimizer or sequence-length settings.',
   'Revalidate independent auditor input/source identities and create new test/readiness receipts before a v2 attempt.',
 ],
 'test_gap':'The v1 synthetic runtime fixture stored the same synthetic config dict in both locations, so it did not represent the archived checkpoint schema.',
 'module_sha256':module_hashes,'input_sha256':inputs,
}
# Verify every original byte identity again before persisting the separate diagnosis.
for relative,digest in inputs.items():
    assert sha(ROOT/relative)==digest
write(OUT/'diagnosis.json',report)
print(json.dumps({'complete':True,'failed_predicates':[k for k,v in checks.items() if not v],
 'train_config_exact':True,'diagnosis_sha256':sha(OUT/'diagnosis.json'),
 'input_count':len(inputs),'cuda_initialized':torch.cuda.is_initialized()}))
