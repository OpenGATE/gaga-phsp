import torch.nn as nn
import torch
from box import Box

import gaga_phsp as gaga
from torch import Tensor
import re


class MyLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(MyLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.clamp(x, min=0.0) + torch.clamp(x, max=0.0) * self.negative_slope


def get_activation(params):
    activation = None
    # set activation
    if params["activation"] == "relu":
        activation = nn.ReLU()
    if params["activation"] == "leaky_relu":
        activation = nn.LeakyReLU()
    if params["activation"] == "my_leaky_relu":
        activation = MyLeakyReLU()
    if not activation:
        print("Error, activation unknown: ", params["activation"])
        exit(0)
    return activation


class Discriminator(nn.Module):
    """
    Discriminator: D(x, θD)
    """

    def __init__(self, params):
        super(Discriminator, self).__init__()
        print(f'D IN=x_dim={params["x_dim"]}  d_dim={params["d_dim"]}    OUT=1')
        x_dim = params["x_dim"]
        d_dim = params["d_dim"]
        d_l = params["d_layers"]
        sn = False
        if "spectral_norm" in params:
            sn = params["spectral_norm"]
        # activation function
        activation = get_activation(params)
        # create the net
        self.net = nn.Sequential()
        # first layer
        if sn:
            self.net.add_module(
                "1st_layer", nn.utils.spectral_norm(nn.Linear(x_dim, d_dim))
            )
        else:
            self.net.add_module("1st_layer", nn.Linear(x_dim, d_dim))

        # hidden layers
        for i in range(d_l):
            self.net.add_module(f"activation_{i}", activation)
            if sn:
                self.net.add_module(
                    f"layer_{i}", nn.utils.spectral_norm(nn.Linear(d_dim, d_dim))
                )
            else:
                self.net.add_module(f"layer_{i}", nn.Linear(d_dim, d_dim))
        # latest layer
        if params["loss"] == "non-saturating-bce":
            self.net.add_module("sigmoid", nn.Sigmoid())
        else:
            self.net.add_module("last_activation", activation)
            self.net.add_module("last_layer", nn.Linear(d_dim, 1))

        # for p in self.parameters():
        #    if p.ndimension() > 1:
        #        nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    """
    Generator: G(z, θG) -> x fake samples
    """

    def __init__(self, params):
        super(Generator, self).__init__()
        self.params = params
        print(
            f'G x_dim={params["x_dim"]}  g_dim={params["g_dim"]}  '
            f'z_dim={params["z_dim"]}   cond_dim={len(params["cond_keys"])}'
        )
        # the total input dim for the G is z_dim + conditional_keys (if any)
        z_dim = params["z_dim"] + len(params["cond_keys"])
        x_dim = params["x_dim"] - len(params["cond_keys"])
        g_dim = params["g_dim"]
        g_l = params["g_layers"]
        print(f'G ---> IN=z_dim={z_dim}  g_dim={params["g_dim"]}  OUT=x_dim={x_dim}')
        # activation function
        activation = get_activation(params)
        # create the net
        self.net = nn.Sequential()
        # first layer
        self.net.add_module("first_layer", nn.Linear(z_dim, g_dim))
        # hidden layers
        for i in range(g_l):
            self.net.add_module(f"activation_{i}", activation)
            self.net.add_module(f"layer_{i}", nn.Linear(g_dim, g_dim))
        # last layer
        self.net.add_module(f"last_activation_{g_l}", activation)
        self.net.add_module(f"last_layer", nn.Linear(g_dim, x_dim))

        # initialisation (not sure better than default init). Keep default.
        for p in self.parameters():
            if p.ndimension() > 1:
                nn.init.kaiming_normal_(p)  ## seems better ???
                # nn.init.xavier_normal_(p)
                # nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

    def forward(self, x):
        return self.net(x)

    def forward_with_post_processing(self, x):
        # generate data
        y = self.net(x)
        # denormalize
        y = (y * self.x_std) + self.x_mean
        # apply post_process
        y = self.post_process(y, self.params)
        return y

    def init_forward_with_post_processing(self, f, gpu):
        self.post_process = f
        self.init_forward_with_denorm(gpu)
        self.forward = self.forward_with_post_processing

    def init_forward_with_denorm(self, gpu):
        # init the std/mean in torch variable
        device = gaga.init_pytorch_gpu(gpu, False)
        self.x_mean = Tensor(torch.from_numpy(self.params["x_mean"]).to(device))
        self.x_std = Tensor(torch.from_numpy(self.params["x_std"]).to(device))
        # by default, bypass the test in  denormalization
        # (if forward_with_post_processing is used, the test is kept)
        self.forward = self.forward_with_norm

    def forward_with_norm(self, x):
        y = self.net(x)
        y = (y * self.x_std) + self.x_mean
        return y


class PositionalEncoding:
    """
    Apply positional encoding to the input.
    """

    def __init__(self, options):
        self.num_encoding_functions = options["num_encoding_functions"]
        self.include_input = options["include_input"]
        self.log_sampling = options["log_sampling"]
        self.blocks_description = {}
        self.frequency_bands = None

    def get_number_of_dim(self):
        i = 0
        for k in self.blocks_description:
            if self.blocks_description[k]["pe"]:
                i += (
                    self.blocks_description[k]["length"]
                    * self.num_encoding_functions
                    * 2
                )
                if self.include_input:
                    i += self.blocks_description[k]["length"]
            else:
                i += self.blocks_description[k]["length"]
        return i

    def encode(self, x):
        if self.log_sampling:
            self.frequency_bands = 2.0 ** torch.linspace(
                0.0,
                self.num_encoding_functions - 1,
                self.num_encoding_functions,
                dtype=x.dtype,
                device=x.device,
            )
        else:
            self.frequency_bands = torch.linspace(
                2.0**0.0,
                2.0 ** (self.num_encoding_functions - 1),
                self.num_encoding_functions,
                dtype=x.dtype,
                device=x.device,
            )
        # The input tensor is added to the positional encoding.
        encoding = [x] if self.include_input else []
        for freq in self.frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(x * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)

    def set_dim_to_encode(self, params, keys, start_dim=None):
        self.blocks_description = {}
        ci = 0
        if start_dim is not None:
            self.blocks_description["z"] = {
                "start": 0,
                "length": start_dim,
                "pe": False,
            }
            ci = start_dim
        else:
            start_dim = 0
        pe = params["positional_encoding"]
        if len(keys) > 0:
            keys = keys.split(" ")
        for k in keys:
            idx = keys.index(k)
            if idx < ci - start_dim:
                continue
            if k not in pe:
                self.blocks_description[k] = {
                    "start": ci,
                    "length": 1,
                    "pe": False,
                }
                ci += 1
            else:
                pattern = rf"{k}\s+(\d+)"
                match = re.search(pattern, pe)
                n = int(match.group(1))
                self.blocks_description[k] = {"start": ci, "length": n, "pe": True}
                ci += n
        self.blocks_description = Box(self.blocks_description)
        print("final", self.blocks_description)

    def apply_pos_encoding_to_some_dim(self, x):
        # apply positional encoding to some dimension of the x vector
        x_blocks = [
            (
                self.encode(x[:, b.start : b.start + b.length])
                if b.pe
                else x[:, b.start : b.start + b.length]
            )
            for b in self.blocks_description.values()
        ]
        return torch.cat(x_blocks, dim=-1)


class DiscriminatorPosEncoding(nn.Module):
    """
    Discriminator: D(x, θD)
    """

    def __init__(self, params):
        super(DiscriminatorPosEncoding, self).__init__()
        x_dim = params["x_dim"]
        d_dim = params["d_dim"]
        d_l = params["d_layers"]
        cond_dim = len(params["cond_keys"])

        sn = False
        if "spectral_norm" in params:
            sn = params["spectral_norm"]

        # activation function
        activation = get_activation(params)

        # positional encoding
        self.pos_encoder = PositionalEncoding(params["positional_encoding_options"])
        keys = params["keys"]
        self.pos_encoder.set_dim_to_encode(params, keys)
        pe_dim = self.pos_encoder.get_number_of_dim()

        print(f"Discriminator {x_dim=}")
        print(f"Discriminator {keys=}")
        print(f"Discriminator {cond_dim=}")
        print(f"Discriminator {pe_dim=} X {d_dim=}")

        # create the net
        self.net = nn.Sequential()
        # first layer
        if sn:
            # self.net.add_module('1st_layer', nn.utils.spectral_norm(nn.Linear(x_dim, d_dim)))
            self.net.add_module(
                "1st_layer", nn.utils.spectral_norm(nn.Linear(pe_dim, d_dim))
            )
        else:
            # self.net.add_module('1st_layer', nn.Linear(x_dim, d_dim))
            self.net.add_module("1st_layer", nn.Linear(pe_dim, d_dim))

        # hidden layers
        for i in range(d_l):
            self.net.add_module(f"activation_{i}", activation)
            if sn:
                self.net.add_module(
                    f"layer_{i}", nn.utils.spectral_norm(nn.Linear(d_dim, d_dim))
                )
            else:
                self.net.add_module(f"layer_{i}", nn.Linear(d_dim, d_dim))
        # latest layer
        if params["loss"] == "non-saturating-bce":
            self.net.add_module("sigmoid", nn.Sigmoid())
        else:
            self.net.add_module("last_activation", activation)
            self.net.add_module("last_layer", nn.Linear(d_dim, 1))

        # for p in self.parameters():
        #    if p.ndimension() > 1:
        #        nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

    def forward(self, x):
        x_encoded = self.pos_encoder.apply_pos_encoding_to_some_dim(x)
        return self.net(x_encoded)


class GeneratorPosEncoding(nn.Module):
    """
    Generator: G(z, θG) -> x fake samples
    """

    def __init__(self, params):
        super(GeneratorPosEncoding, self).__init__()
        self.params = params

        # the total input dim for the G is z_dim + conditional_keys (if any)
        self.cond_dim = len(params["cond_keys"])
        self.z_dim = params["z_dim"]  # + cond_dim
        input_dim = self.z_dim + self.cond_dim
        x_dim = params["x_dim"] - self.cond_dim
        g_dim = params["g_dim"]
        g_l = params["g_layers"]

        # activation function
        activation = get_activation(params)

        # create the net
        self.net = nn.Sequential()

        # positional encoding
        print("GeneratorPosEncoding")
        self.pos_encoder = PositionalEncoding(params["positional_encoding_options"])
        keys = " ".join(params["cond_keys"])
        self.pos_encoder.set_dim_to_encode(params, keys, self.z_dim)
        pe_dim = self.pos_encoder.get_number_of_dim()

        print(f"Generator {self.z_dim=}")
        print(f"Generator {self.cond_dim=}")
        print(f"Generator {input_dim=}")
        print(f"Generator {pe_dim=} X {g_dim=}")

        # first layer
        # self.net.add_module('first_layer', nn.Linear(z_dim, g_dim))
        self.net.add_module("first_layer", nn.Linear(pe_dim, g_dim))

        # hidden layers
        for i in range(g_l):
            self.net.add_module(f"activation_{i}", activation)
            self.net.add_module(f"layer_{i}", nn.Linear(g_dim, g_dim))

        # last layer
        self.net.add_module(f"last_activation_{g_l}", activation)
        self.net.add_module(f"last_layer", nn.Linear(g_dim, x_dim))

        # initialisation (not sure better than default init). Keep default.
        for p in self.parameters():
            if p.ndimension() > 1:
                nn.init.kaiming_normal_(p)  ## seems better ???
                # nn.init.xavier_normal_(p)
                # nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

    def forward(self, x):
        # print(f'G {x.shape=}')
        # x = self.pos_encoder(x)

        """z = x[:, : self.z_dim]  # [batch_size, z_dim] latent space Z vector
        event_pos = x[:, self.z_dim : self.z_dim + 3]  # [batch_size, 3] Event position
        event_dir = x[:, self.z_dim + 3 :]  # [batch_size, 3] Event direction
        event_pos_encoded = self.pos_encoder(event_pos)
        event_dir_encoded = self.pos_encoder(event_dir)
        # print(f'G {z.shape=}')
        # print(f'G {event_pos.shape=}')
        # print(f'G {event_dir.shape=}')
        # print(f'G {event_pos_encoded.shape=}')
        # print(f'G {event_dir_encoded.shape=}')
        x_encoded = torch.cat([z, event_pos_encoded, event_dir_encoded], dim=-1)
        # print(f'G {x_encoded.shape=}')"""

        x_encoded = self.pos_encoder.apply_pos_encoding_to_some_dim(x)

        return self.net(x_encoded)
