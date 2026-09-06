export FORCE_TQDM_PROGRESS=1
mkdir -p ./logs

# V8: 多种子 bagging — 2 seeds × 4 折 = 8 模型, GPU 每次并发 4 个, 两波
for f in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 nohup bash -c "python run.py $f; echo \"fold$f EXIT:\$?\"" > ./logs/fold$f.log 2>&1 &
  sleep 10
done
wait
for f in 5 6 7 8; do
  CUDA_VISIBLE_DEVICES=0 nohup bash -c "python run.py $f; echo \"fold$f EXIT:\$?\"" > ./logs/fold$f.log 2>&1 &
  sleep 10
done
wait
echo "ALL 8 FOLDS DONE"
