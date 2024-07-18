#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import json
import click
import colorama
import gatetools.phsp as phsp
from colorama import Fore, Style
import gaga_phsp as gaga
from box import Box
import socket

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "phsp_filename", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.argument(
    "json_filename", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option(
    "--output", "-o", help="Output filename, default = automatic name", default="auto"
)
@click.option(
    "--output_folder",
    "-f",
    help='Output folder (ignored if output is not "auto")',
    default=".",
)
@click.option("--progress-bar/--no-progress-bar", default=True)
@click.option(
    "--user_param_str",
    "-ps",
    help="overwrite str parameter of the json file",
    multiple=True,
    type=(str, str),
)
@click.option(
    "--user_param",
    "-p",
    help="overwrite numeric parameter of the json file",
    multiple=True,
    type=(str, float),
)
@click.option(
    "--user_param_int",
    "-pi",
    help="overwrite numeric int parameter of the json file",
    multiple=True,
    type=(str, int),
)
def gaga_train(
    phsp_filename,
    json_filename,
    output,
    output_folder,
    progress_bar,
    user_param_str,
    user_param,
    user_param_int,
):
    """
    \b
    Train GAN to learn a PHSP (Phase Space File)

    \b
    <PHSP_FILENAME>   : input PHSP file (.npy)
    <JSON_FILENAME>   : input json file with all GAN parameters
    """

    # term color
    colorama.init()

    # read parameters and convert to box for simpler access
    params_file = open(json_filename).read()
    params = Box(json.loads(params_file))

    # overwrite params from the command line
    gaga.update_params_with_user_options(params, user_param)
    gaga.update_params_with_user_options(params, user_param_str)
    gaga.update_params_with_user_options(params, user_param_int)

    # read input training dataset
    print(Fore.CYAN + "Loading training dataset ... " + phsp_filename + Style.RESET_ALL)
    x, read_keys, m = phsp.load(phsp_filename)

    # special processing for 'keys' (required)
    gaga.param_check_keys(params, read_keys)

    # consider only selected keys
    params.params_filename = json_filename
    x = phsp.select_keys(x, read_keys, params.keys_list)

    # add information in the params
    params.training_size = len(x)
    params.x_dim = len(params.keys_list)
    params.progress_bar = progress_bar
    params.training_filename = phsp_filename
    start = datetime.datetime.now()
    params.start_date = start.strftime(gaga.date_format)
    params.hostname = socket.gethostname()

    # check and init parameters (required)
    gaga.check_input_params(params)

    # print parameters
    for e in params:
        if e[0] != "#":
            print(f"   {e:32s} {params[e]}")

    # build the model
    print(Fore.CYAN + "Building the GAN model ..." + Style.RESET_ALL)
    gan = gaga.Gan(params)

    # train
    print(Fore.CYAN + "Start training ..." + Style.RESET_ALL)
    # model = gan.train(x)
    model = gan.train2(x)

    # stop timer
    stop = datetime.datetime.now()
    params.end_date = stop.strftime(gaga.date_format)

    # prepare output filename
    gaga.auto_output_filename(params, output, output_folder)

    # save output (params is in the model)
    gan.save(model, params.output_filename)

    print(Fore.CYAN + f"Training done: {params.output_filename}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    gaga_train()
