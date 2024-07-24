

# Test 001 non conditional GAN

- Generate gaussian 
- train
- generate samples from trained and compare
- around 1 min

# Test 002 conditional GAN

- Generate gaussian 
- train
- generate samples from trained and compare
- around 1 min

# Test 003 

Convert exit position to ideal position

# Test 004

SPECT Intevo
- main1: reference simulation 1e6 Bq
- main2: with garf only 2e5 Bq
- main3: with garf and gaga
- main4: standalone with numpy
- main5: standalone with torch

Timing (07/2024)   linux (nvidia)        osx (mps)
- main1 8 threads  6.8min PPS=64,706     2.5min PPS=174,523    
- main2 8 threads  1.2min PPS=69,885      37sec PPS=215,687
- main3 1 thread    43sec PPS=138,160     24sec PPS=243,391
- main4 1 thread    22sec PPS=233,992     23sec PPS=226,310
- main5 1 thread     6sec PPS=994,562     WRONG mps???