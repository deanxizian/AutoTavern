"""Best-effort once-only exception forensic snapshot. Never changes sampling decisions."""
import sys,os,json,time,pickle,gzip,hashlib,traceback,random
from pathlib import Path
import numpy as np

def capture_actor_failure(sampler,identity,selected,full,actor_observations,pending,counts,ready,active,seats,error):
    context=getattr(sampler,'failure_capture_context',None)
    if context is None:return {'status':'capture_context_not_configured_original_exception_preserved'}
    folder=Path(context['output_directory'])
    try:
        folder.mkdir(parents=True,exist_ok=True)
        reservation=folder/'capture-reservation.json'
        try:
            with reservation.open('x')as stream:json.dump({'unix':time.time(),'pid':os.getpid(),'identity':str(identity),'original_error':repr(error)},stream)
        except FileExistsError:return {'status':'prior_capture_preserved_no_overwrite'}
        from capture_support import load
        normalizer=load('native_state').normalize
        profile=load('profile_tools')
        slots=np.asarray(selected,dtype=np.int64).copy();roles=int(sampler.roles)
        games=sorted({int(slot//roles)for slot in slots})
        states={};errors=[]
        for game in games:
            try:states[str(game)]=sampler.batch.save(game)
            except Exception as exc:errors.append({'game':game,'native_save_error':repr(exc)})
        def shaper(value):return normalizer({k:v for k,v in vars(value).items()if k!='layout'})
        histories={}
        for game in games:
            histories[str(game)]={'learner':shaper(sampler.shapers[game])if sampler.shapers else None,'opponents':{str(slot):shaper(value)for slot,value in sampler.opponent_shapers.items()if slot//roles==game}}
        torch=sys.modules.get('torch');include_cuda=bool(torch is not None and torch.cuda.is_initialized())
        rng=(profile.capture_rng(torch,include_cuda,[torch.cuda.current_device()]if include_cuda else None)if torch is not None else {'python_random':random.getstate(),'numpy_legacy':normalizer(np.random.get_state()),'torch_not_imported':True,'CUDA_not_initialized':True})
        immutable_context={k:context[k]for k in ['policy_sources','configuration','source_bindings','sampling_origin']if k in context}
        payload={'schema':'actor-exception-forensics071-v1','identity':str(identity),'selected_slots':slots,'selected_games':np.asarray([s//roles for s in slots],np.int64),'selected_roles':np.asarray([s%roles for s in slots],np.int64),'selected_physical_seats':np.asarray(seats)[slots].copy(),'physical_seats_all':np.asarray(seats).copy(),'role_names':np.asarray(sampler.role_names).copy(),'full_observations':normalizer(full),'actor_observations':normalizer(actor_observations),'ready_filter':normalizer(ready),'active_games':normalizer(active),'counts':normalizer(counts),'pending':normalizer(pending),'Native_states':states,'game_shaper_histories':histories,'episodes':normalizer(sampler.episodes),'episode_steps':normalizer(sampler.episode_steps),'restore_versions':normalizer(sampler._restore_versions),'RNG':rng,'context':normalizer(immutable_context),'exception':{'type':type(error).__name__,'message':str(error),'repr':repr(error),'traceback':traceback.format_exc()},'capture_errors':errors,'excluded':['actor/model objects and GPU graph','static shaper.layout fixed by sourceSHA'],'CUDA_RNG_included_only_if_already_initialized':include_cuda}
        normalized=normalizer(payload);digest=profile.typed_digest(normalized)
        artifact=folder/'failure-snapshot.pkl.gz'
        with artifact.open('xb')as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0)as stream:pickle.dump(normalized,stream,protocol=5)
        masks=np.asarray(actor_observations['action_mask'],dtype=bool)
        summary={'status':'captured_actor_exception_original_will_be_reraised','identity':str(identity),'selected_slots':slots.tolist(),'empty_mask_local_rows':np.flatnonzero(~masks.any(axis=-1)).tolist(),'selected_games':games,'artifact':str(artifact),'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'typed_digest':digest,'exception_type':type(error).__name__,'exception_message':str(error),'capture_errors':errors,'CUDA_RNG_included':include_cuda,'model_or_actor_graph_serialized':False}
        (folder/'capture-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary
    except BaseException as capture_error:
        # This must never replace the original actor exception. No retry or policy fallback.
        failure={'status':'capture_failed_original_exception_preserved','capture_error':repr(capture_error),'original_error':repr(error)}
        try:(folder/'capture-error.json').write_text(json.dumps(failure,indent=2)+'\n')
        except BaseException:print('forensic capture failed: '+repr(capture_error),file=sys.stderr)
        return failure
