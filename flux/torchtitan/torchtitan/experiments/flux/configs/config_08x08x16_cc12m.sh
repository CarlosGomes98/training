export MOUNT_CURRENT_DIR=True;
export CONFIG_FILE=./torchtitan/experiments/flux/train_configs/flux_schnell_model.toml;
export CONT=../containers/optimized+torchtitan.flux.7;
export LOGDIR=./schnell_8n_gbs_1024_30k_cc12m;
export CHECKPOINT_FREQ=30000;
export DATASET=cc12m-wds-disk;
export PARAMS="--lr_scheduler.warmup_steps=3000 \
--metrics.log_freq=10 \
--parallelism.data_parallel_replicate_degree=1 \
--parallelism.data_parallel_shard_degree=-1 \
--training.batch_size=16 \
--training.dataset_path= \
--eval.eval_freq=600"
