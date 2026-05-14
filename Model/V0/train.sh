mkdir -p ./logs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
CUDA_VISIBLE_DEVICES=$((0 % GPU_COUNT)) nohup python run.py 1 > ./logs/fold1.log 2>&1 &
CUDA_VISIBLE_DEVICES=$((1 % GPU_COUNT)) nohup python run.py 2 > ./logs/fold2.log 2>&1 &
CUDA_VISIBLE_DEVICES=$((2 % GPU_COUNT)) nohup python run.py 3 > ./logs/fold3.log 2>&1 &
CUDA_VISIBLE_DEVICES=$((3 % GPU_COUNT)) nohup python run.py 4 > ./logs/fold4.log 2>&1 &
wait
echo "All four folds done."