@echo off
REM V1 training — 4-fold parallel training (Windows)
if not exist logs mkdir logs
set GPU_COUNT=1
for /f %%i in ('nvidia-smi -L 2^>nul ^| find /c "UUID"') do set GPU_COUNT=%%i
set /a DEV0=0 %% GPU_COUNT
set /a DEV1=1 %% GPU_COUNT
set /a DEV2=2 %% GPU_COUNT
set /a DEV3=3 %% GPU_COUNT
start "f1" /B cmd /c "set FORCE_TQDM_PROGRESS=1&& set CUDA_VISIBLE_DEVICES=%DEV0%&& python run.py 1 > logs\fold1.log 2>&1"
start "f2" /B cmd /c "set FORCE_TQDM_PROGRESS=1&& set CUDA_VISIBLE_DEVICES=%DEV1%&& python run.py 2 > logs\fold2.log 2>&1"
start "f3" /B cmd /c "set FORCE_TQDM_PROGRESS=1&& set CUDA_VISIBLE_DEVICES=%DEV2%&& python run.py 3 > logs\fold3.log 2>&1"
start "f4" /B cmd /c "set FORCE_TQDM_PROGRESS=1&& set CUDA_VISIBLE_DEVICES=%DEV3%&& python run.py 4 > logs\fold4.log 2>&1"
