import os
# Starts 8 experiments under 8 separate screens on 8 GPUs.

datasets = ["Copy-v0", "DuplicatedInput-v0", "ReversedAddition-v0", "ReversedAddition3-v0"]

os.system("rm logs_*")
os.system("k trpo_rnn_")
os.system("screen -wipe")


for dataset in datasets:
    os.system("screen -dm -S trpo_rnn_%s bash -c '. ~/.profile; ~/.bashrc; CUDA_VISIBLE_DEVICES=[] python main.py %s 2>&1 | tee logs_%s ; bash'" % (dataset, dataset, dataset))
