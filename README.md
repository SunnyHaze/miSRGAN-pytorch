# Prepare Datas
```
python .\generate_sr_volumes.py \
    --data_path="G:\Datasets\Medical\datas\manifest-gJIZVVFt6412408718812805737\PROSTATE-DIAGNOSIS"
    --                                                
```
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
train_main.py \
    --world_size 2 \
    --data_path "/root/Dataset/prostate/meta_data_revised.json" \
    --batch_size 8 \
    --epochs 70 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 2 \
    --output_dir ./output_dir_2/ \
    --log_dir ./output_dir_2/ \
2> train_error_2.log 1>train_logs_2.log
