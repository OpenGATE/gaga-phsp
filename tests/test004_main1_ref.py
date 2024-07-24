#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gaga_phsp.spect_intevo_helpers import *
from gaga_phsp.gaga_helpers_tests import get_tests_folder
import opengate.tests.utility as utility

if __name__ == "__main__":
    # units
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second
    deg = gate.g4_units.deg

    output_folder = get_tests_folder() / "output" / "test004"
    data_folder = get_tests_folder() / "data" / "test004"

    # spect options
    simu = SpectIntevoSimulator('monte_carlo', "test004_main1_ref")
    simu.output_folder = output_folder
    simu.ct_image = data_folder / "53_CT_bg_crop_4mm_vcrop.mhd"
    simu.activity_image = data_folder / "three_spheres_4mm.mhd"
    simu.radionuclide = "tc99m"
    simu.head_angles = [0 * deg, 100 * deg, 230 * deg]

    # simu.visu = True
    simu.duration = 30 * sec
    simu.number_of_threads = 8
    simu.total_activity = 1e6 * Bq

    simu.image_size = [96, 96]
    simu.image_spacing = [4.7951998710632 * mm * 3, 4.7951998710632 * mm * 3]

    # init the simulation
    sim = gate.Simulation()
    sim.seed = 123456
    simu.create_gate_simulation(sim)

    # go
    sim.run()

    # print results at the end
    stats = sim.output.get_actor("stats")
    print(stats)

    # compare results
    is_ok = test_check_results(simu, stats, data_folder / "test004_ref", simu.name, [65, 20])
    utility.test_ok(is_ok)
