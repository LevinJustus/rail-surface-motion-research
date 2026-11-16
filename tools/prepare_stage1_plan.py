"""Freeze first-training subsets, order and settings; never open images or train.

Python 3.8+ standard library only. Re-running --check verifies byte-for-byte
reproducibility against the existing source split and model configuration.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261114
SUBSET_VERSION = 'pilot_subset_v1'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def canonical(value):
    return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')

def balanced_subset(rows, count, partition, split_sha):
    groups = collections.defaultdict(list)
    for row in rows:
        assert row['assigned_split']==partition
        groups[row['group_id']].append(row)
    prefix=f'{SEED}|{SUBSET_VERSION}|{split_sha}|{partition}'
    order=sorted(groups,key=lambda g:(digest(f'{prefix}|group|{g}'.encode()),g))
    for group in order:
        groups[group]=sorted(groups[group],key=lambda r:(digest(f'{prefix}|image|{r["id"]}'.encode()),r['id']))
    chosen=[];level=0
    while len(chosen)<count:
        before=len(chosen)
        for group in order:
            if level<len(groups[group]):
                chosen.append(groups[group][level])
                if len(chosen)==count:break
        assert len(chosen)>before,'Insufficient unique candidates'
        level+=1
    return [{k:r[k] for k in ['id','group_id','image','mask','assigned_split']} for r in chosen]

def make_schedule(name, rows, epochs, batch_size):
    updates=[]
    for epoch in range(1,epochs+1):
        ordered=sorted([r['id'] for r in rows],key=lambda rid:(
            digest(f'{SEED}|stage1_epoch_v1|{name}|{epoch}|{rid}'.encode()),rid))
        assert len(ordered)%batch_size==0
        for start in range(0,len(ordered),batch_size):
            updates.append({'update':len(updates)+1,'epoch':epoch,
                            'sample_ids':ordered[start:start+batch_size]})
    return updates

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true',help='Verify existing files; do not write')
    args=parser.parse_args()
    source_path=ROOT/'data/splits/conservative_group_v1.json'
    source_bytes=source_path.read_bytes();split_sha=digest(source_bytes)
    assert split_sha==(ROOT/'data/splits/conservative_group_v1.sha256').read_text().strip()
    source=json.loads(source_bytes)
    model_path=ROOT/'configs/baseline.json';model_sha=digest(model_path.read_bytes())
    train=[r for r in source['records'] if r['assigned_split']=='training']
    val=[r for r in source['records'] if r['assigned_split']=='validation']
    assert len(train)==1396 and len(val)==400
    assert not {r['group_id'] for r in train}&{r['group_id'] for r in val}
    pilot_train=balanced_subset(train,128,'training',split_sha)
    pilot_val=balanced_subset(val,64,'validation',split_sha)
    smoke=pilot_train[:8]
    assert len({r['group_id'] for r in smoke})==8
    chosen_train={r['id'] for r in pilot_train};chosen_val={r['id'] for r in pilot_val}
    assert len(chosen_train)==128 and len(chosen_val)==64 and not chosen_train&chosen_val
    subsets={
      'protocol':SUBSET_VERSION,'selection_seed':SEED,'source_split_sha256':split_sha,
      'selection_rule':'Hash-ranked groups and IDs, round robin without replacement; smoke is first8 of pilot training',
      'group_rank_key':'SHA256(seed|pilot_subset_v1|split_sha|partition|group|group_id)',
      'image_rank_key':'SHA256(seed|pilot_subset_v1|split_sha|partition|image|id)',
      'image_access':'Metadata only; no image, mask or model output used for selection',
      'smoke_train':smoke,'pilot_train':pilot_train,'pilot_validation':pilot_val}
    subset_bytes=canonical(subsets);subset_sha=digest(subset_bytes)
    schedules={'protocol':'stage1_epoch_v1','seed':SEED,'subset_sha256':subset_sha,
       'ordering_key':'SHA256(seed|stage1_epoch_v1|profile|epoch|id)',
       'replacement':False,'batch_size':2,
       'smoke':make_schedule('smoke',smoke,4,2),
       'pilot':make_schedule('pilot',pilot_train,3,2),
       'validation_order':[r['id'] for r in pilot_val]}
    schedule_bytes=canonical(schedules);schedule_sha=digest(schedule_bytes)
    assert len(schedules['smoke'])==16 and len(schedules['pilot'])==192
    for profile,rows in [('smoke',smoke),('pilot',pilot_train)]:
        expected={r['id'] for r in rows}
        for epoch in sorted({s['epoch'] for s in schedules[profile]}):
            visited=[rid for s in schedules[profile] if s['epoch']==epoch for rid in s['sample_ids']]
            assert len(visited)==len(expected) and set(visited)==expected
    config={
      'schema_version':1,'plan_id':'stage1_training_v1','status':'planned_not_started',
      'scope':'Step07 protocol only; smoke is Step08, pilot is Step09',
      'model_config':'configs/baseline.json','model_config_sha256':model_sha,
      'source_split':'data/splits/conservative_group_v1.json','source_split_sha256':split_sha,
      'subsets':'data/splits/pilot_subset_v1.json','subsets_sha256':subset_sha,
      'batch_schedule':'data/splits/stage1_batch_schedule_v1.json','batch_schedule_sha256':schedule_sha,
      'execution_environment':'cpu','precision':'single','fresh_initialization_per_profile':True,
      'model_initialization_seed':SEED,'selection_and_order_seed':SEED,
      'augmentation':'none','image_preprocessing':'existing aspect-preserving downsampling and valid-mask padding',
      'normalization':'uint8 to single/255 once; no dataset-fitted statistics',
      'jpeg_label_threshold':128,'output_channel_order':['background','defect'],
      'optimizer':{'name':'Adam','learning_rate':0.001,'beta1':0.9,'beta2':0.999,
        'epsilon':1e-8,'learning_rate_schedule':'constant','weight_decay':0.0,
        'gradient_clipping':'global_L2_norm','max_gradient_norm':1.0},
      'loss':{'name':'masked_BCE_softDice','bce_weight_positive_target':0.5,
        'dice_weight_positive_target':0.5,'empty_target':'BCE only at weight1.0',
        'bce_numerics':'stable log-softmax from two pre-softmax logits; no second sigmoid',
        'dice_smoothing':1e-6,'spatial_reduction':'valid pixels per image',
        'batch_reduction':'mean of per-image loss','empty_valid_region':'error, never silently skip'},
      'profiles':{
        'smoke':{'training_subset':'smoke_train','unique_train_images':8,'batch_size':2,
          'passes':4,'target_optimizer_updates':16,'validation':'training set diagnostics only',
          'evaluation_updates':[0,4,8,12,16],'checkpoint_updates':[0,8,16],
          'require_generalization_score':False,'require_dice_threshold':False},
        'pilot':{'training_subset':'pilot_train','validation_subset':'pilot_validation',
          'unique_train_images':128,'unique_validation_images':64,'batch_size':2,
          'passes':3,'target_optimizer_updates':192,'validation_updates':[0,64,128,192],
          'checkpoint_every_updates':32,'save_initial':True,'save_final':True,
          'early_stopping':False,'primary_report':'last checkpoint plus complete validation history',
          'save_best_validation':True,'best_rule':'highest micro foreground IoU; tie lower valid loss; then earlier update'}},
      'metrics':{'foreground_threshold':0.5,'threshold_comparison':'>=',
        'postprocessing':'none','padding':'excluded from all confusion counts',
        'required':['micro_foreground_IoU','micro_foreground_Dice','micro_precision','micro_recall',
          'macro_image_foreground_IoU','macro_group_foreground_IoU','predicted_foreground_fraction',
          'target_foreground_fraction','per_image_TP_FP_FN_TN'],
        'zero_denominator':'NaN, report denominator and undefined count; never silently convert to1',
        'all_background_prediction':'Report explicitly; not hidden by pixel accuracy',
        'validation_statistics':'descriptive only; pilot not final test or significance evidence'},
      'visualization':{'fixed_train_ids':[r['id'] for r in smoke],
        'fixed_validation_ids':[r['id'] for r in pilot_val[:8]],
        'panels':['input','ground_truth','defect_probability','thresholded_prediction'],
        'selection':'fixed before training; no best-looking sample selection'},
      'checkpoints':{'directory':'runs/<run_id>/checkpoints',
        'required':['network_learnables_and_state','optimizer_first_and_second_moments',
          'global_update','epoch','next_schedule_position','rng_state',
          'model_config_and_hash','training_config_and_hash','subset_and_schedule_hash',
          'source_split_hash','git_commit','MATLAB_environment','class_channel_order','elapsed_time'],
        'resume':'same profile/config/hash only; retain optimizer iteration and next batch',
        'prediction_reload_max_abs_error':1e-6},
      'logging':{'per_update':['total_loss','BCE_component','Dice_component','unclipped_global_gradient_norm',
          'clip_factor','update_time_s','sample_ids','valid_pixel_count'],
        'per_validation':['full fixed64 metrics','per-image confusion counts','validation_time_s'],
        'training_data_access':'training subset only; validation inference only; no heldout test access'},
      'failure_policy':{'nonfinite_loss_or_gradient':'stop and record failed update; never count a failed update as success',
        'data_or_hash_mismatch':'stop before training',
        'interrupted_run':'save last consistent checkpoint; do not mark step complete',
        'slow_CPU':'time smoke first, record pilot estimate before starting; any budget change requires new config version',
        'poor_learning':'diagnose with training/validation only; retain this version and create a new plan for changed hyperparameters'}}
    cfg_bytes=canonical(config)
    check={'status':'PASS','training_started':False,'source_split_sha256':split_sha,
       'model_config_sha256':model_sha,'subset_sha256':subset_sha,'schedule_sha256':schedule_sha,
       'training_config_sha256':digest(cfg_bytes),'smoke_images':8,'smoke_groups':8,
       'pilot_train_images':128,'pilot_train_groups':len({r['group_id'] for r in pilot_train}),
       'pilot_validation_images':64,'pilot_validation_groups':len({r['group_id'] for r in pilot_val}),
       'smoke_updates':16,'pilot_updates':192,'raw_images_opened':0,
       'heldout_test_images_opened':0,'train_validation_group_overlap':0,
       'checks':['source hash','unique subset IDs','subset containment','group separation',
                 'exact per-pass coverage','update counts','validation order fixed']}
    outputs={
      'data/splits/pilot_subset_v1.json':subset_bytes,
      'data/splits/pilot_subset_v1.sha256':(subset_sha+'\n').encode(),
      'data/splits/stage1_batch_schedule_v1.json':schedule_bytes,
      'data/splits/stage1_batch_schedule_v1.sha256':(schedule_sha+'\n').encode(),
      'configs/training_stage1_v1.json':cfg_bytes,
      'configs/training_stage1_v1.sha256':(digest(cfg_bytes)+'\n').encode(),
      'docs/evidence/stage1_plan_check.json':canonical(check)}
    for name,payload in outputs.items():
        path=ROOT/name
        if args.check:
            assert path.exists() and path.read_bytes()==payload,'Plan differs: '+name
        else:
            if path.exists():assert path.read_bytes()==payload,'Refusing to overwrite a different frozen plan: '+name
            else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    print(json.dumps(check,ensure_ascii=False,indent=2))
    print('STAGE1_PLAN_CHECK_OK' if args.check else 'STAGE1_PLAN_FROZEN; NO TRAINING RUN')

if __name__=='__main__':main()
