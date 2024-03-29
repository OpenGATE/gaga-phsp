# (Test 1 : Linac phsp)

==> See GateBenchmark/t9_gaga_phsp

- test train
- test convert from pth to pt, with and without denormalisation

# Test 2: Gaussian

(need to save npy folder before)

generate:

    cd tests
    gaga_gauss_test npy/gauss_v1.npy -n 1e5 -t v1
    gaga_gauss_test npy/gauss_v2.npy -n 1e5 -t v2

train:

    gaga_train npy/gauss_v1.npy json/g1.json -f pth/ -pi epoch 1000
    gaga_train npy/gauss_v2.npy json/g2.json -f pth/ -pi epoch 5000

result:

    gaga_gauss_plot npy/gauss_v1.npy pth/g1_GP_SquareHinge_1_1000.pth -n 1e4
    gaga_plot  npy/gauss_v1.npy pth/g1_GP_SquareHinge_1_1000.pth

    gaga_gauss_plot npy/gauss_v2.npy pth/g2_GP_SquareHinge_1_5000.pth -n 1e4
    gaga_plot  npy/gauss_v2.npy pth/g2_GP_SquareHinge_1_5000.pth

# Test 3: pairs parametrisation

The initial data was obtained with (phsp sphere of 210 mm): 

    gaga_pet_to_pairs pet_iec.root -o pairs.npy -n 1e4

Convert pairs to tlor then convert back to pairs: 

    gaga_pairs_to_tlor npy/pairs.npy -o npy/tlor.npy
    gaga_tlor_to_pairs npy/tlor.npy -o npy/pairs2.npy -r 210
    gt_phsp_plot npy/pairs.npy npy/pairs2.npy

==> See GateBenchmark/t14_phsp_pairs


# Test 4 : conditional Gaussian

generate:

    ../bin/gaga_gauss_cond_test -n 1e6 npy/xgauss_10_1e6.npy -m 10

train:
    
    gaga_train npy/xgauss_10_1e6.npy json/cg1.json -f pth -pi epoch 4000

result:
    
    # warning x and y not independent here! 
    gaga_gauss_plot npy/xgauss_10_1e6.npy pth/cg1_GP_SquareHinge_1_4000.pth -n 1e5 -x 3.5 -m 1e4 -y 1.16666


# Test 5 : spect parametrisation

Convert root dataset to parametrisation, replace P_exit by P_ideal = P_exit - c x t x Dir 

    gaga_exit_pos_to_ideal_pos spect_training_dataset.root -n 1e4 -o a.npy
    gaga_ideal_pos_to_exit_pos a.npy -n 1e4 -o b.npy
    gt_phsp_plot b.npy spect_training_dataset.root -n 1e4


# Test 6 : TODO gaga_garf_generate_img