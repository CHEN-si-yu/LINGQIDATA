export FORCE_TQDM_PROGRESS=1
mkdir -p ./logs

# CUDA_VISIBLE_DEVICES=0 nohup python run.py 1 > ./logs/fold1.log 2>&1 &
# pid=$!
# wait $pid
# CUDA_VISIBLE_DEVICES=0 nohup python run.py 2 > ./logs/fold2.log 2>&1 &
# pid=$!
# wait $pid
# CUDA_VISIBLE_DEVICES=0 nohup python run.py 3 > ./logs/fold3.log 2>&1 &
# pid=$!
# wait $pid
CUDA_VISIBLE_DEVICES=0 nohup python run.py 4 > ./logs/fold4.log 2>&1 &
pid=$!
wait $pid
