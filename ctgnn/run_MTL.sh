#! /bin/bash
echo "I'm begining training!"
/home/slj108/miniconda3/envs/qh/bin/python MTL_Trainer.py --precision 16 --batch_size 128 --max_epochs 30 --accelerator ddp --gnn_head GCN --gnn_residual --adj_mat_path ./adj_all_65/adj_binary.npy --valid_tasks defect water --task_weight Fixed --task_weights_fixed 27 1 --use_auxilliary --progress_bar_refresh_rate 2 --flush_logs_every_n_steps 100 --log_every_n_steps 100  --log_save_dir ./log 
