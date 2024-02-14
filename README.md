# Summary 
Final project for GP-GPU course at École nationale supérieure d'informatique et de mathématiques appliquées de Grenoble

# How to run
1. Move project code to GPU_REMOTE
```
rsync -avz --exclude '.git' --exclude 'benchmark_data.csv' ../* GPU_REMOTE
```
2. On remote, run
```
python benchmark_implementation.py TYPE FILE
```
 where
   - `TYPE` - 'sequential' or 'parallelized' -- version of program which you want to benchmark,
   - `FILE` - file name to which measured time should be stored.

Script `benchmark_implementation.py` will generate synthetic data of different sizes, execute selected `TYPE` of the algorithm on it and store measured times to the `FILE`.
