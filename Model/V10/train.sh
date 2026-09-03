export FORCE_TQDM_PROGRESS=1
mkdir -p ./logs

# V10: 三风格集成 — 保守(1-4) / 激进(5-8) / 平衡(9-12), GPU 每次并发 4 个, 三波
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
for f in 9 10 11 12; do
  CUDA_VISIBLE_DEVICES=0 nohup bash -c "python run.py $f; echo \"fold$f EXIT:\$?\"" > ./logs/fold$f.log 2>&1 &
  sleep 10
done
wait
echo "ALL 12 FOLDS DONE"
