export PYTHONPATH=/workspace/lightx2v:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 test_acc.py
