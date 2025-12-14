python3 ./imitate_episodes.py \
--task_name bend \
--ckpt_dir ./ckpt/bend \
--policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --eval --backbone resnet18 --temporal_agg \
--num_epochs 1 --lr 1e-5 \
--seed 0