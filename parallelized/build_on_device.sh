CC=nvcc
GPU=vmgpu034
OUT_NAME=gemm_cuda

rsync -avz --exclude '.git' ../* forostib@vmgpu034.ensimag.fr:~/project
ssh forostib@vmgpu034.ensimag.fr 'bash -l -c "cd project/parallelized && pwd && ls && make --always-make && ./mixture_models ../data.csv 3 10 2 means sigma 100 "'
scp forostib@vmgpu034.ensimag.fr:~/project/parallelized/sigma forostib@vmgpu034.ensimag.fr:~/project/parallelized/means .
