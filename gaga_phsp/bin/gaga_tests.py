#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import pathlib
from pathlib import Path
import click
import random
import colored
from gaga_phsp.gaga_helpers_tests import color_ok, color_error, get_tests_folder

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--test_id", "-i", default="all", help="Start test from this number")
@click.option(
    "--random_tests",
    "-r",
    is_flag=True,
    default=False,
    help="Start the last 10 tests and 1/4 of the others randomly",
)
def go(test_id, random_tests):
    mypath = get_tests_folder()
    print(f"Looking for tests in: {mypath}")

    ignored_tests = [
        "test045_speedup",  # this is a binary (still work in progress)
    ]

    onlyfiles = [
        f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))
    ]

    files = []
    for f in onlyfiles:
        if "wip" in f:
            print(f"Ignoring: {f:<40} ")
            continue
        if "visu" in f:
            continue
        if "OLD" in f:
            continue
        if "old" in f:
            continue
        if "test" not in f:
            continue
        if ".py" not in f:
            continue
        if ".log" in f:
            continue
        if "all_tes" in f:
            continue
        if "_base" in f:
            continue
        if "_helpers" in f:
            continue
        if os.name == "nt" and "_mt" in f:
            continue
        if f in ignored_tests:
            continue
        files.append(f)

    files = sorted(files)
    if test_id != "all":
        test_id = int(test_id)
        files_new = []
        for f in files:
            id = int(f[4:7])
            if id >= test_id:
                files_new.append(f)
            else:
                print(f"Ignoring: {f:<40} (< {test_id}) ")
        files = files_new
    elif random_tests:
        files_new = files[-10:]
        prob = 0.25
        files = files_new + random.sample(files[:-10], int(prob * (len(files) - 10)))
        files = sorted(files)

    print(f"Running {len(files)} tests (warning ~1min tests)")
    print(f"-" * 70)

    failure = False

    for f in files:
        start = time.time()
        print(f"Running: {f:<46}  ", end="")
        cmd = "python " + os.path.join(mypath, f"{f}")
        log = os.path.join(os.path.dirname(mypath), Path(mypath) / "log" / f"{f}.log")
        r = os.system(f"{cmd} > {log} 2>&1")
        # subprocess.run(cmd, stdout=f, shell=True, check=True)
        if r == 0:
            print(colored.stylize(" OK", color_ok), end="")
        else:
            if r == 2:
                # this is probably a Ctrl+C, so we stop
                print("Stopped by user")
                exit(-1)
            else:
                print(colored.stylize(" FAILED !", color_error), end="")
                failure = True
                os.system("cat " + log)
        end = time.time()
        print(f"   {end - start:5.1f} s     {log:<65}")

    print(not failure)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
