## retrain the searched model
nohup python -u -m torch.distributed.launch --nproc_per_node=8 --master_port 9960 retrain.py \
            --batch_size=1024 \
            --epochs=240 \
            --seed=0 \
            --save='eval_models' \
            --data= \
            --model_id='' \
            > log.txt 2>&1 &


## valid the retrained model
# python valid.py --checkpoint= \
#                 --data= \
#                 --model_id=''
