rsync -avz --exclude '.git' --exclude 'benchmark_data.csv' ../* forostib@vmgpu034.ensimag.fr:~/project
ssh forostib@vmgpu034.ensimag.fr 'bash -l -c "cd project/parallelized && pwd && ls && make --always-make && ./mixture_models ../data.csv 3 10 2 means sigma 100 gpu_time.csv"'
scp forostib@vmgpu034.ensimag.fr:~/project/parallelized/sigma forostib@vmgpu034.ensimag.fr:~/project/parallelized/means .
