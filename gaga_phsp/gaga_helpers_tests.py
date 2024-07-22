import os
import inspect
import colored
import sys
import scipy
import numpy as np
import pathlib

try:
    color_error = colored.fg("red") + colored.attr("bold")
    color_warning = colored.fg("orange_1")
    color_ok = colored.fg("green")
except AttributeError:
    # new syntax in colored>=1.5
    color_error = colored.fore("red") + colored.style("bold")
    color_warning = colored.fore("orange_1")
    color_ok = colored.fore("green")


def fatal(s):
    caller = inspect.getframeinfo(inspect.stack()[1][0])
    ss = f"(in {caller.filename} line {caller.lineno})"
    ss = colored.stylize(ss, color_error)
    print(ss)
    s = colored.stylize(s, color_error)
    print(s)
    raise Exception(s)


def run_and_check(cmd):
    print()
    print(f'Running : {cmd}')
    r = os.system(f"{cmd} ")
    if r != 0:
        fatal(f"Command error : {cmd}")


def test_ok(is_ok=False):
    if is_ok:
        s = "Great, tests are ok."
        s = "\n" + colored.stylize(s, color_ok)
        print(s)
        # sys.exit(0)
    else:
        s = "Error during the tests !"
        s = "\n" + colored.stylize(s, color_error)
        print(s)
        sys.exit(-1)


def compare_sampled_points(keys, real, fake, wtol=0.1, tol=0.08):
    for i in range(len(keys)):
        w = scipy.stats.wasserstein_distance(real[:, i], fake[:, i])
        print(f"({i}) Key {keys[i]}, wass = {w:.2f}  tol = {wtol:.2f}")
        if w > wtol:
            fatal(f"Difference between real and fake too large {w} vs {wtol}")
        real_mean = np.mean(real[:, i])
        real_std = np.std(real[:, i])
        fake_mean = np.mean(fake[:, i])
        fake_std = np.std(fake[:, i])
        d_mean = np.fabs((real_mean - fake_mean) / real_mean)
        d_std = np.fabs((real_std - fake_std) / real_std)
        print(f"({i}) Mean real vs fake : {real_mean:.2f} {fake_mean:.2f} {d_mean * 100:.2f}%")
        print(f"({i}) Std real vs fake : {real_std:.2f} {fake_std:.2f} {d_std * 100:.2f}%")
        if d_mean > tol or d_std > tol:
            fatal(f"Difference between real and fake too large {d_mean} {d_std} vs {tol}")


def get_tests_folder():
    p = pathlib.Path(__file__).parent.resolve()
    p = os.path.abspath(p / ".." / "tests")
    return pathlib.Path(p)
