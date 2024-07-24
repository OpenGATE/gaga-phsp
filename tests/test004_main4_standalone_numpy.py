#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gaga_phsp.spect_intevo_helpers import *
from gaga_phsp.gaga_helpers_tests import get_tests_folder

if __name__ == "__main__":
    # units
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second
    deg = gate.g4_units.deg

    output_folder = get_tests_folder() / "output" / "test004"
    data_folder = get_tests_folder() / "data" / "test004"

    # spect options
    simu = SpectIntevoSimulator('standalone_numpy', "test004_main4_standalone_numpy")
    simu.output_folder = output_folder
    simu.ct_image = data_folder / "53_CT_bg_crop_4mm_vcrop.mhd"  # (needed to position the source)
    simu.activity_image = data_folder / "three_spheres_4mm.mhd"
    simu.radionuclide = "tc99m"
    simu.gantry_angles = [0 * deg, 100 * deg, 230 * deg]

    simu.duration = 30 * sec
    simu.number_of_threads = 1
    simu.total_activity = 2e5 * Bq
    # simu.visu = True

    simu.image_size = [96, 96]
    simu.image_spacing = [4.7951998710632 * mm * 3, 4.7951998710632 * mm * 3]

    simu.gaga_source.pth_filename = data_folder / "train_gaga_v100_GP_0GP_20.0_200.pth"
    simu.garf_detector.pth_filename = data_folder / "train_arf_intevo_tc99m_lehr_data304_v043_v7.pth"
    simu.garf_detector.hit_slice_flag = False
    simu.gaga_source.batch_size = 1e5
    simu.gaga_source.backward_distance = 600 * mm

    # run the simulation
    simu.generate_projections()

    # print
    print(f"Time = {simu.computation_time_duration:0.1f} seconds ")
    print(f"PPS = {simu.pps} ")

    # compare results
    is_ok = test_check_results(simu, None,
                               data_folder / "test004_ref",
                               "test004_main3_gaga",
                               [87, 15],
                               scaling=1)
    utility.test_ok(is_ok)
