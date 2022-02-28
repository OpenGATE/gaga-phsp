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

    gaga_train npy/gauss_v1.npy json/g1.json -f pth/ -pi epoch 1000
    gaga_train npy/gauss_v2.npy json/g2.json -f pth/ -pi epoch 5000

result:

    gaga_gauss_plot npy/gauss_v1.npy pth/g1_GP_SquareHinge_1_1000.pth -n 1e4
    gaga_plot  npy/gauss_v1.npy pth/g1_GP_SquareHinge_1_1000.pth

    gaga_gauss_plot npy/gauss_v2.npy pth/g2_GP_SquareHinge_1_5000.pth -n 1e4
    gaga_plot  npy/gauss_v2.npy pth/g2_GP_SquareHinge_1_5000.pth

# Test 3: pairs parametrisation

==> See GateBenchmark/t14_phsp_pairs


# Test 4 : conditional Gaussian

generate:

    ../bin/gaga_gauss_cond_test -n 1e6 npy/xgauss_10_1e6.npy -m 10

train:
    
    gaga_train npy/xgauss_10_1e6.npy json/cg1.json -f pth -pi epoch 4000

result:
    
    # warning x and y not independent here! 
    gaga_gauss_plot npy/xgauss_10_1e6.npy pth/cg1_GP_SquareHinge_1_4000.pth -n 1e5 -x 3.5 -m 1e4 -y 1.16666
