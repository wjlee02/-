python3 ./imitate_episodes.py \
--task_name box \
--ckpt_dir ./ckpt/box \
--policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 --backbone resnet18 \
--num_epochs 500 --lr 1e-5 --data_folders original -1 dark -1 light -1 \
--seed 0