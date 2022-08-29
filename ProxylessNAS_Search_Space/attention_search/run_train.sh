## train attention_supernet
nohup python -u -m torch.distributed.launch --nproc_per_node=8 train.py \
            --epochs=120 \
            --batch_size=512 \
            --epochs=120 \
            --seed=0 \
            --data= \
            > logdir/log_att_supernet 2>&1 &


## eval attention weights
# python -u -m torch.distributed.launch --nproc_per_node=1 compute_attention.py \
#             --seed=0 \
#             --data= \
#             --pretrained_path=''


## export searched model
# python export_op.py
