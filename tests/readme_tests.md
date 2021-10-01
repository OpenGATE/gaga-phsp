# Test 1 : Linac phsp

==> See GateBenchmark/t9_gaga_phsp
- test train
- test convert from pth to pt, with and without denormalisation

# Test 2: Gaussian
   
generate:

    cd tests
    gaga_gauss_test npy/gauss_v1.npy -n 1e5 -t v1
    gaga_gauss_test npy/gauss_v2.npy -n 1e5 -t v2

train:

    gaga_train npy/gauss_v1.npy json/g1.json -f . -pi epoch 1000
    gaga_train npy/gauss_v2.npy json/g2.json -f . -pi epoch 5000
    
result:

    gaga_gauss_plot npy/gauss_v1.npy pth/g1_GP_SquareHinge_1_1000.pth -n 1e4
    gaga_plot  npy/gauss_v1.npy pth/g1_GP_SquareHinge_1_1000.pth

    gaga_gauss_plot npy/gauss_v2.npy pth/g2_GP_SquareHinge_1_5000.pth -n 1e4
    gaga_plot  npy/gauss_v2.npy pth/g2_GP_SquareHinge_1_5000.pth

# Test 3: pairs parametrisation

==> See GateBenchmark/t14_phsp_pairs


