


# Files organisation

`gaga_model`

Definition of the Discriminator and the Generator nets.

`gaga_trainer`
`gaga_helpers`


- gaga_penalty ?
- gaga_plot_helpers
- LOGGER


# Options and parameters

| loss_type               | wasserstein non-saturating-bce                                   |
| penalty_type            | zero_penalty clamp_penalty gradient_penalty gradient_penalty_max |
| penalty_weight          |                                                                  |
| layer_norm              | true/false                                                       |
| d_layers g_layers       |                                                                  |
| d_dim g_dim             |                                                                  |
| z_dim                   |                                                                  |
| clamp_lower clamp_upper |                                                                  |
|-------------------------+------------------------------------------------------------------|
| optimiser               | adam RMSprop                                                     |
| shuffle                 | true/false                                                       |

    "#": "adam optimiser: regularisation L2 (for adam only) ; zero if no regul",
    "d_weight_decay": 0.5,
    "g_weight_decay": 0.5,

    "#": "Real and Fake instance Gaussian noise sigma. -1 for none. ",
    "r_instance_noise_sigma": 0,
    "f_instance_noise_sigma": 0,  

    "#": "adam optimiser: beta",
    "beta_1": "0.9",
    "beta_2": "0.999",

    "#": "optimiser: learning rate",
    "d_learning_rate": 1e-4,
    "g_learning_rate": 1e-4,

    "#": "optimiser: number of D and G update by epoch",
    "d_nb_update": 2,
    "g_nb_update": 1,

    "#": "optimiser: max nb of epoch (iteration)",
    "epoch": 80000,

    "#": "optimiser: nb of samples by batch",
    "batch_size": 1000,

    "#": "Smooth fake/real labels instead of zero/one",
    "#label_smoothing": 0.2,

    "#": "---------------------------------------------------------------------",
    "#": " DATA ",
    "#": "---------------------------------------------------------------------",
    "keys": "X dY",
        
    "#": "---------------------------------------------------------------------",
    "#": " GENERAL ",
    "#": "---------------------------------------------------------------------",
        
    "#": "gpu_mode: true false auto",
    "gpu_mode": "auto",

    "#": "save Generator and info every epoch",
    "dump_epoch_start": 0,
    "dump_epoch_every": 5000,
    "dump_last_n_epoch": 0

}
