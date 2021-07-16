



# Test 1 : Linac phsp

Linac phase space 
Trainnig dataset : `ELEKTA_PRECISE_6mv_part1.root`

Todo in `GateBenchmark/t9_gaga_phsp`

Training 
    
    conda activate pet_gan 
    gaga_train2 train_data/ELEKTA_PRECISE_6mv_part1.root pth2/config_001.json -o pth2/001.pth
    gaga_plot train_data/ELEKTA_PRECISE_6mv_part1.root pth2/001.pth -n 1e4
    gaga_convert_pth_to_pt pth2/001.pth

 


Gate simulation

    conda activate gate
    gate_devel
    
    # in GateBenchmarks, using 001.pth
    ./runTest.sh 

    # or manually
    Gate -a "[JOB_ID,0][N,1e2][TYPE,gaga][PTH,003]" mac/main_test.mac

    # or tests
    gate_split_and_run.py mac/main_test.mac -a N 1e7 -a PTH 001 -a TYPE gaga -j 1 -o output_001
    gate_split_and_run.py mac/main_test.mac -a N 1e7 -a PTH 002 -a TYPE gaga -j 1 -o output_002
    gate_split_and_run.py mac/main_test.mac -a N 1e7 -a PTH 003 -a TYPE gaga -j 1 -o output_003


Quantitative test

    ./runAnalysis.py output
    
    ./runAnalysis.py output_001/output.local_0
    ./runAnalysis.py output_002/output.local_0
    ./runAnalysis.py output_003/output.local_0
    


# Test 2: Gaussian

    
generate:

    cd tests
    gaga_gauss_test gauss_v1.npy -n 1e5 -t v1
    gaga_gauss_test gauss_v3.npy -n 1e5 -t v3

train:

    gaga_train gauss_1.npy g1.json -f . -pi epoch 100
    

result:

    gaga_gauss_plot gauss_v1.npy g1_GP_SquareHinge_1_100.pth -n 1e4
    gaga_plot  gauss_v1.npy g1_GP_SquareHinge_1_100.pt

