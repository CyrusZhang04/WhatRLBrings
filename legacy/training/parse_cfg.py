# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import ast, sys, re, json
def extract(path):
    txt=open(path,errors='replace').read()
    i=txt.find("{'actor_rollout_ref'")
    if i<0:
        i=txt.find("{'algorithm'")
    if i<0: return None, txt
    # find matching brace
    depth=0; j=i; instr=None; esc=False
    while j<len(txt):
        c=txt[j]
        if instr:
            if esc: esc=False
            elif c=='\\': esc=True
            elif c==instr: instr=None
        else:
            if c in ('"',"'"): instr=c
            elif c=='{': depth+=1
            elif c=='}':
                depth-=1
                if depth==0: break
        j+=1
    s=txt[i:j+1]
    try:
        d=ast.literal_eval(s)
    except Exception as e:
        return ('ERR',str(e)), txt
    return d, txt[j+1:]
def g(d,path,default='NA'):
    cur=d
    for p in path.split('.'):
        if isinstance(cur,dict) and p in cur: cur=cur[p]
        else: return default
    return cur
KEYS=['actor_rollout_ref.model.path','actor_rollout_ref.actor.optim.lr','actor_rollout_ref.actor.optim.lr_warmup_steps','actor_rollout_ref.actor.optim.weight_decay','actor_rollout_ref.actor.optim.warmup_style',
'actor_rollout_ref.actor.ppo_mini_batch_size','actor_rollout_ref.actor.ppo_epochs','actor_rollout_ref.actor.use_kl_loss','actor_rollout_ref.actor.kl_loss_coef','actor_rollout_ref.actor.kl_loss_type','actor_rollout_ref.actor.clip_ratio_low','actor_rollout_ref.actor.clip_ratio_high','actor_rollout_ref.actor.clip_ratio_c','actor_rollout_ref.actor.loss_agg_mode','actor_rollout_ref.actor.entropy_coeff','actor_rollout_ref.actor.policy_loss.loss_mode','actor_rollout_ref.actor.ulysses_sequence_parallel_size','actor_rollout_ref.actor.ppo_max_token_len_per_gpu','actor_rollout_ref.actor.grad_clip',
'actor_rollout_ref.rollout.name','actor_rollout_ref.rollout.n','actor_rollout_ref.rollout.temperature','actor_rollout_ref.rollout.top_p','actor_rollout_ref.rollout.response_length','actor_rollout_ref.rollout.prompt_length','actor_rollout_ref.rollout.tensor_model_parallel_size','actor_rollout_ref.rollout.gpu_memory_utilization','actor_rollout_ref.rollout.val_kwargs','actor_rollout_ref.rollout.mode','actor_rollout_ref.rollout.max_num_batched_tokens',
'algorithm.adv_estimator','algorithm.use_kl_in_reward','algorithm.kl_ctrl','algorithm.norm_adv_by_std_in_grpo','algorithm.filter_groups',
'data.train_files','data.val_files','data.train_batch_size','data.val_batch_size','data.max_prompt_length','data.max_response_length','data.prompt_key','data.filter_overlong_prompts','data.truncation','data.shuffle','data.gen_batch_size',
'reward_model.reward_manager','reward_model.overlong_buffer','custom_reward_function',
'trainer.project_name','trainer.experiment_name','trainer.total_epochs','trainer.total_training_steps','trainer.save_freq','trainer.test_freq','trainer.nnodes','trainer.n_gpus_per_node','trainer.default_local_dir','trainer.logger','trainer.resume_mode','trainer.val_before_train','trainer.rollout_data_dir','trainer.validation_data_dir','trainer.max_actor_ckpt_to_keep','trainer.critic_warmup']
if __name__=='__main__':
    for p in sys.argv[1:]:
        d,rest=extract(p)
        print('#####',p)
        if d is None: print('  no cfg'); continue
        if isinstance(d,tuple): print('  parse err',d); continue
        for k in KEYS:
            v=g(d,k)
            if v!='NA': print('  ',k,'=',v)
