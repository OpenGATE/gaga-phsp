#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import opengate.contrib.spect.siemens_intevo as gate_intevo
import opengate.tests.utility as utility
import gaga_phsp as gaga
import garf
from scipy.spatial.transform import Rotation
from pathlib import Path
import numpy as np
import os
import time
import itk
from scipy.ndimage import map_coordinates


class SpectIntevoSimulator:
    """
    Class to create different types of SPECT simulation with the Intevo model

    Modes:
    - "monte_carlo": create a gate SPECT simulation. Can be with or without gaga or garf
    - "training_gaga":
    - "training_garf":
    - "standalone_numpy":
    - "standalone_torch":

    TODO
    - volume name in gate are fixed, should be based on the global name
    - no check nor explicit parameters requirements

    """

    def __init__(self, mode, name=None):
        # units
        Bq = gate.g4_units.Bq
        sec = gate.g4_units.second
        mm = gate.g4_units.mm
        mm = gate.g4_units.mm

        self.mode = mode
        available_modes = [
            "training_gaga",  # only ct, no spect head nor garf
            "training_garf",  # only garf
            "monte_carlo",  # with gate, can be with or without gaga or garf
            "standalone_numpy",  # out of gate, use numpy (for mac with mps GPU)
            "standalone_torch",
        ]  # out of gate, use torch (fastest, need cuda GPU)
        if mode not in available_modes:
            gate.exception.fatal(
                f"Error, mode {mode} not available. " f"Known modes: {available_modes}"
            )

        self.output_folder = Path("output")
        self.name = name
        if name is None:
            self.name = "spect_simulator"
        self.visu = False

        self.ct_image = None
        self.activity_image = None
        self.radionuclide = "tc99m"

        self.total_activity = 1e4 * Bq
        self.duration = 1 * sec
        self.number_of_threads = 1
        self.number_of_events = None
        self.branching_ratio = None

        # gaga
        self.gaga_source = gaga.GagaSource()

        # garf
        self.garf_detector = garf.GarfDetector()

        # for multiple heads (gate only, not standalone)
        self.head_angles = [0]

        # Projection angles (and corresponding rotation)
        self.gantry_angles = [0]
        self.gantry_rotations = None

        # gantry distance
        self.radius = 380 * mm
        self.detector_offset = 0 * mm  # (not used yet)
        self.collimator_type = None  # auto set with radionuclide

        # image info will be copied to garf
        self.image_size = [256, 256]
        self.image_spacing = [4.7951998710632 * mm / 2, 4.7951998710632 * mm / 2]

        # for production cut
        self.cut_in_head = 1 * mm
        self.cut_in_ct = 4 * mm

        # for training_gaga
        self.sphere_phsp_radius = 610 * mm

        # may be computed
        self.start_time = None
        self.stop_time = None
        self.computation_time_duration = None
        self.pps = None
        self.output_filenames = []

    def initialize(self):
        # FIXME check required parameters wrt to mode

        # collimator
        if self.collimator_type is None:
            self.collimator_type = get_collimator_from_radionuclide(self.radionuclide)

        # output
        self.initialize_output()

        # number of events
        self.initialize_nb_events()

        # gantry rotation
        self.initialize_gantry_rotations()

        # garf
        self.set_garf_parameters()

        # gaga source
        self.set_gaga_parameters()

    def initialize_nb_events(self):
        Bq = gate.g4_units.Bq
        sec = gate.g4_units.second
        w, e = gate.sources.generic.get_rad_gamma_energy_spectrum(self.radionuclide)
        self.branching_ratio = np.sum(w)
        ne = int(
            (self.total_activity / Bq) * self.branching_ratio * self.duration / sec
        )
        self.number_of_events = ne

    def initialize_gantry_rotations(self):
        deg = gate.g4_units.deg
        self.gantry_rotations = []
        for angle in self.gantry_angles:
            r = Rotation.from_euler("z", angle / deg, degrees=True)
            self.gantry_rotations.append(r)

    def set_garf_parameters(self):
        if not self.garf_is_used():
            return
        # crystal distance if arf is used
        _, d, _ = gate_intevo.compute_plane_position_and_distance_to_crystal(
            self.collimator_type
        )

        # parameters
        self.garf_detector.radius = self.radius
        self.garf_detector.crystal_distance = d
        self.garf_detector.image_size = np.array(self.image_size.copy())
        self.garf_detector.image_spacing = np.array(self.image_spacing.copy())

        # rotation wrt Intevo
        # r1 = rotation like the detector (needed), see in digitizer ProjectionActor
        r1 = Rotation.from_euler("yx", (90, 90), degrees=True)
        # r2 = rotation X180 is to set the detector head-foot, rotation Z90 is the gantry angle
        r2 = Rotation.from_euler("xz", (180, 90), degrees=True)
        r = r2 * r1
        self.garf_detector.initial_plane_rotation = r

    def set_gaga_parameters(self):
        # activity file
        self.gaga_source.activity_filename = self.activity_image
        # translation bw ct and activity images
        tr = gate.image.get_translation_between_images_center(
            self.ct_image, self.activity_image
        )
        self.gaga_source.cond_translation = tr

    def initialize_output(self):
        # output folder ?
        os.makedirs(self.output_folder, exist_ok=True)
        # filenames
        self.output_filenames = []
        for i in range(len(self.gantry_angles)):
            n = self.output_folder / f"{self.name}_proj_{i}.mhd"
            self.output_filenames.append(n)

    def garf_is_used(self):
        if self.garf_detector.pth_filename is not None:
            return True
        return False

    def dump_number_of_gammas(self):
        Bq = gate.g4_units.Bq
        sec = gate.g4_units.second
        apt = self.total_activity / self.number_of_threads
        ne = self.number_of_events
        print(f"Source {self.radionuclide} yield: {self.branching_ratio}")
        print(f"Source duration: {self.duration / sec:.2f} sec")
        print(f"Source total activity: {self.total_activity / Bq:.0f} Bq")
        print(f"Source total activity per thread: {apt / Bq:.0f} Bq")
        print(f"Expected nb of gammas: {ne}")
        print(f"Expected nb of gammas per thread: {ne / self.number_of_threads:.0f}")
        return ne

    def create_gate_simulation(self, sim):
        # check mode
        m = ["training_gaga", "training_garf", "monte_carlo"]
        if self.mode not in m:
            gate.exception.fatal(
                f"Cannot create a gate simulation with this mode "
                f"{self.mode}, use one of {m}"
            )

        # mode training gaga : no spect head, only CT and spherical phsp
        if self.mode == "training_gaga":
            # force some parameters, there are unused
            self.gaga_source.pth_filename = None
            self.head_angles = []
            self.phsp = add_gaga_sphere_phsp(sim, self.sphere_phsp_radius)
            self.phsp.output_filename = f"{self.name}.root"

        # init and check param
        self.initialize()

        # default options
        sim.check_volumes_overlap = True
        sim.visu_type = "qt"
        sim.number_of_threads = self.number_of_threads
        sim.output_dir = self.output_folder
        sim.progress_bar = True

        # visu ?
        if self.visu:
            Bq = gate.g4_units.Bq
            sim.number_of_threads = 1
            self.total_activity = 1 * Bq
            sim.visu = True

        # volumes
        ct = None
        if self.gaga_source.pth_filename is None:
            if sim.visu:
                ct = add_fake_ct_image(sim, self.ct_image, cut=self.cut_in_ct)
            else:
                ct = add_ct_image(sim, self.ct_image, cut=self.cut_in_ct)

        # spect head
        deg = gate.g4_units.deg
        heads = []
        for head_angle in self.head_angles:
            print(f"Add a SPECT head with angle {head_angle / deg:.0f} deg")
            if self.garf_detector.pth_filename is None:
                h = add_intevo_head(
                    sim,
                    self.collimator_type,
                    self.radius,
                    self.detector_offset,
                    angle=head_angle,
                    cut=self.cut_in_head,
                )
            else:
                h = add_intevo_arf_plane(
                    sim,
                    self.collimator_type,
                    self.radius,
                    self.detector_offset,
                    head_angle,
                    self.image_size,
                    self.image_spacing,
                )

            heads.append(h)

        # physics
        m = gate.g4_units.m
        sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
        sim.physics_manager.set_production_cut("world", "all", 1e3 * m)

        # source
        if self.gaga_source.pth_filename is None:
            source = add_vox_gamma_source(
                sim,
                self.activity_image,
                self.radionuclide,
                self.total_activity,
                self.duration,
                ct,
                self.ct_image,
            )
        else:
            source = add_gaga_source(
                sim,
                self.ct_image,
                self.activity_image,
                self.total_activity,
                self.gaga_source.pth_filename,
                self.gaga_source.backward_distance,
                self.gaga_source.batch_size,
            )

        # stat actors
        stats = sim.add_actor("SimulationStatisticsActor", "stats")

        # digitizer
        projections = []
        for head in heads:
            if self.garf_detector.pth_filename is None:
                p = add_intevo_digitizer(
                    sim, head, self.radionuclide, self.image_size, self.image_spacing
                )
            else:
                p = add_intevo_arf_actor(
                    sim,
                    self.collimator_type,
                    head,
                    self.garf_detector.pth_filename,
                    self.image_size,
                    self.image_spacing,
                    self.garf_detector.gpu_mode,
                    self.garf_detector.batch_size,
                )
            projections.append(p)

        # output
        stats.output_filename = f"{self.name}_stats.txt"
        if len(heads) == 1:
            projections[0].output_filename = f"{self.name}_proj.mhd"
        else:
            i = 0
            for proj in projections:
                proj.output_filename = f"{self.name}_proj_{i}.mhd"
                i += 1

        # timing
        sim.run_timing_intervals = [[0, self.duration]]
        self.dump_number_of_gammas()

    def generate_projections(self):
        # special case with standalone : no head_angle, use gantry_angles
        if len(self.head_angles) != 1:
            gate.exception.fatal(
                f"Cannot use several head angles in standalone. Use gantry_angles"
            )

        # mode ?
        # initialize : set all parameters to garf and gaga
        self.initialize()

        # initialize garf and gaga
        self.garf_detector.initialize(self.gantry_rotations)
        self.gaga_source.initialize()

        # verbose
        print(self.garf_detector)
        print()
        print(self.gaga_source)

        # go
        images = []
        self.start_time = time.time()
        if self.mode == "standalone_numpy":
            images = self.gaga_source.generate_projections_numpy(
                self.garf_detector, self.number_of_events
            )
        if self.mode == "standalone_torch":
            images = self.gaga_source.generate_projections_torch(
                self.garf_detector, self.number_of_events
            )

        # timing
        self.stop_time = time.time()
        self.computation_time_duration = self.stop_time - self.start_time
        self.pps = int(self.number_of_events / self.computation_time_duration)

        # write images
        i = 0
        for image in images:
            output = self.output_filenames[i]
            itk.imwrite(image, output)
            print(f"Done, saving output in {output}")
            i += 1


def get_collimator_from_radionuclide(radionuclide):
    rad_to_colli = {"tc99m": "lehr", "lu177": "melp"}
    radionuclide = radionuclide.lower()
    if radionuclide not in rad_to_colli:
        gate.exception.fatal(
            f"Unknown radionuclide {radionuclide}. "
            f"Valid radionuclides are = {rad_to_colli}"
        )
    return rad_to_colli[radionuclide]


def add_ct_image(sim, ct_image, tol=None, voxels_materials=None, cut=None):
    ct = sim.add_volume("Image", "ct")
    ct.material = "G4_AIR"
    ct.image = ct_image
    gcm3 = gate.g4_units.g_cm3
    # materials
    if voxels_materials is None:
        f1 = gate.utility.get_data_folder() / "Schneider2000MaterialsTable.txt"
        f2 = gate.utility.get_data_folder() / "Schneider2000DensitiesTable.txt"
        if tol is None:
            tol = 0.05 * gcm3
        vm, materials = gate.geometry.materials.HounsfieldUnit_to_material(
            sim, tol, f1, f2
        )
        ct.voxel_materials = vm
    else:
        ct.voxel_materials = voxels_materials

    # phys
    mm = gate.g4_units.mm
    if cut is None:
        cut = 4 * mm
    sim.physics_manager.set_production_cut(ct.name, "all", cut)

    # verbose
    img_info = gate.image.read_image_info(ct_image)
    print(f"CT image: {ct.image}")
    print(f"CT image size (pixels): {img_info.size}")
    print(f"CT image size (mm): {img_info.size * img_info.spacing}")
    print(f"CT image spacing: {img_info.spacing}")
    print(f"CT image translation: {ct.translation}")
    print(f"CT image tol: {tol / gcm3} g/cm3")
    print(f"CT image mat: {len(ct.voxel_materials)} materials")

    return ct


def add_fake_ct_image(sim, ct_image, cut=None):
    ct = sim.add_volume("Box", "ct")
    ct.material = "G4_WATER"
    img_info = gate.image.read_image_info(ct_image)
    ct.size = img_info.size * img_info.spacing

    # phys
    mm = gate.g4_units.mm
    if cut is None:
        cut = 4 * mm
    sim.physics_manager.set_production_cut(ct.name, "all", cut)

    # verbose
    print(f"CT FAKE image: {ct_image}")
    print(f"CT FAKE image size (pixels): {img_info.size}")
    print(f"CT FAKE image size (mm): {img_info.size * img_info.spacing}")
    print(f"CT FAKE image spacing: {img_info.spacing}")
    print(f"CT FAKE image translation: {ct.translation}")

    return ct


def add_gaga_sphere_phsp(sim, radius):
    mm = gate.g4_units.mm
    sim.add_parallel_world("sphere_world")
    sph_surface = sim.add_volume("Sphere", "phase_space_sphere")
    sph_surface.rmin = radius * mm
    sph_surface.rmax = radius * mm + 0.01 * mm
    sph_surface.color = [0, 1, 0, 1]
    sph_surface.material = "G4_AIR"
    sph_surface.mother = "sphere_world"

    # phsp
    phsp = sim.add_actor("PhaseSpaceActor", "phase_space")
    phsp.attached_to = "phase_space_sphere"
    phsp.attributes = [
        "KineticEnergy",
        "PrePosition",
        "PreDirection",
        "TimeFromBeginOfEvent",
        "EventID",
        "EventKineticEnergy",
        "EventPosition",
        "EventDirection",
    ]
    # this option allow to store all events even if absorbed
    phsp.store_absorbed_event = True
    f = sim.add_filter("ParticleFilter", "f")
    f.particle = "gamma"
    phsp.filters.append(f)
    return phsp


def add_intevo_head(sim, colli_type, radius, detector_offset, angle=0, cut=None):
    # create head
    deg = gate.g4_units.deg
    name = f"head_{angle / deg:.0f}"
    head, colli, crystal = gate_intevo.add_spect_head(
        sim, name, colli_type, debug=sim.visu
    )
    # initial default orientation
    rot = gate_intevo.set_head_orientation(head, colli_type, radius, angle / deg)

    # offset
    head.translation[2] = detector_offset

    # cut
    mm = gate.g4_units.mm
    if cut is None:
        cut = 1 * mm
    sim.physics_manager.set_production_cut(head.name, "all", cut)

    return head


def add_intevo_arf_plane(
    sim, colli_type, radius, detector_offset, angle, image_size, image_spacing
):
    # garf plane
    deg = gate.g4_units.deg
    name = f"arf_plane_{angle / deg:.0f}"
    plane_size = [image_size[0] * image_spacing[0], image_size[1] * image_spacing[1]]
    arf_plane = gate_intevo.add_detection_plane_for_arf(
        sim, plane_size, colli_type, radius, angle / deg, name
    )
    # table offset
    arf_plane.translation[2] = detector_offset
    return arf_plane


def add_intevo_arf_actor(
    sim,
    colli_type,
    detector_plane,
    pth_filename,
    image_size,
    image_spacing,
    gpu_mode="auto",
    batch_size=1e5,
):
    # get crystal dist
    pos, crystal_dist, psd = gate_intevo.compute_plane_position_and_distance_to_crystal(
        colli_type
    )
    # arf actor
    arf = sim.add_actor("ARFActor", f"arf_{detector_plane.name}")
    arf.attached_to = detector_plane.name
    arf.batch_size = batch_size
    arf.image_size = image_size
    arf.image_spacing = image_spacing
    arf.verbose_batch = False  # FIXME
    arf.distance_to_crystal = crystal_dist
    arf.gpu_mode = gpu_mode
    arf.enable_hit_slice = False
    arf.pth_filename = pth_filename
    print(f"ARF {arf.name} batch size: {arf.batch_size}")
    print(f"ARF {arf.name} distance_to_crystal: {arf.distance_to_crystal:.2f} mm")
    print(f"ARF {arf.name} pth: {pth_filename}")
    return arf


def add_vox_gamma_source(
    sim, activity_image, radionuclide, total_activity, duration, ct, ct_image
):
    source = sim.add_source("VoxelsSource", "vox_source")
    w, e = gate.sources.generic.get_rad_gamma_energy_spectrum(radionuclide)
    if ct is not None:
        source.attached_to = ct.name
    source.particle = "gamma"
    source.energy.type = "spectrum_lines"
    source.energy.spectrum_weight = w
    source.energy.spectrum_energy = e
    source.image = activity_image
    source.direction.type = "iso"
    if ct is not None:
        source.position.translation = gate.image.get_translation_between_images_center(
            ct_image, source.image
        )
    # set activity
    source.activity = total_activity / sim.number_of_threads
    print(f"Vox source translation: {source.position.translation}")
    return source


def add_intevo_digitizer(sim, head, radionuclide, image_size, image_spacing):
    digit_name = f"{head.name}_digit"
    digit = None
    crystal = sim.volume_manager.get_volume(f"{head.name}_crystal")
    radionuclide = radionuclide.lower()
    if radionuclide == "lu177":
        digit = gate_intevo.add_digitizer_lu177(sim, crystal.name, digit_name)
    if radionuclide == "tc99m":
        digit = gate_intevo.add_digitizer_tc99m(sim, crystal.name, digit_name)
    if digit is None:
        gate.exception.fatal(
            f"Digitizer for radionuclide {radionuclide} is not known (lu177 tc99m)"
        )
    # projection output
    proj = digit.get_last_module()
    proj.size = image_size
    proj.spacing = image_spacing
    return proj


def add_gaga_source(
    sim,
    ct_image,
    activity_image,
    total_activity,
    gaga_pth_filename,
    gaga_backward_distance,
    gaga_batch_size,
):
    # source
    keV = gate.g4_units.keV
    source = sim.add_source("GANSource", "source")
    source.particle = "gamma"
    source.pth_filename = gaga_pth_filename
    source.position_keys = ["PrePosition_X", "PrePosition_Y", "PrePosition_Z"]
    # if p.ideal_pos_flag:
    #    source.position_keys = ["IdealPosition_X","IdealPosition_Y","IdealPosition_Z"]
    #
    source.backward_distance = gaga_backward_distance
    source.direction_keys = ["PreDirection_X", "PreDirection_Y", "PreDirection_Z"]
    source.energy_key = "KineticEnergy"
    source.energy_min_threshold = 10 * keV
    source.skip_policy = "ZeroEnergy"
    source.weight_key = None
    source.time_key = None
    source.backward_force = True  # because we don't care about timing
    source.batch_size = gaga_batch_size
    source.verbose_generator = False  # FIXME
    source.gpu_mode = "auto"

    # activity
    source.activity = total_activity / sim.number_of_threads
    print(f"GAGA Source translation: {source.position.translation}")

    # condition
    cond_gen = gate.sources.gansources.VoxelizedSourceConditionGenerator(activity_image)
    cond_gen.compute_directions = True
    gen = gate.sources.gansources.GANSourceConditionalGenerator(
        source, cond_gen.generate_condition
    )
    source.generator = gen

    # The (source position) conditions are in Geant4 (world) coordinate system
    # (because it is build from phsp Event Position)
    # so the vox sampling should transform the source image points in G4 world
    # This is done by translating from centers of CT and source images
    tr = gate.image.get_translation_between_images_center(ct_image, activity_image)
    print(f"Translation from source coord system to G4 world {tr=} (rotation not done)")
    cond_gen.translation = tr

    return source


def test_check_results(
    simu, stats, ref_simu_folder, simu_name, tols, scaling=1, tol_stat=0.025
):
    ref_simu_folder = Path(ref_simu_folder)
    if stats is not None:
        stats_ref = ref_simu_folder / f"{simu_name}_stats.txt"
        stats_ref = utility.read_stat_file(stats_ref)
        stats.counts.events *= scaling
        stats.counts.tracks *= scaling
        stats.counts.steps *= scaling
        # do not check run counts
        stats.counts.runs = stats_ref.counts.runs
        is_ok = utility.assert_stats(stats, stats_ref, tol_stat)
    else:
        is_ok = True

    n = max(len(simu.head_angles), len(simu.gantry_angles))
    for i in range(n):
        rname = f"{simu_name}_proj_{i}"
        name = f"{simu.name}_proj_{i}"
        is_ok = (
            utility.assert_images(
                ref_simu_folder / f"{rname}.mhd",
                simu.output_folder / f"{name}.mhd",
                stats,
                tolerance=tols[0],
                ignore_value_data1=0,
                ignore_value_data2=0,
                axis="y",
                sum_tolerance=tols[1],
                fig_name=simu.output_folder / f"{name}_{i}.png",
                scaleImageValuesFactor=scaling,
                sad_profile_tolerance=10,
            )
            and is_ok
        )

    return is_ok


def create_1d_profile(image, point1, point2, num_points):
    # Create an array of coordinates along the line
    rows = np.linspace(point1[0], point2[0], num_points)
    cols = np.linspace(point1[1], point2[1], num_points)
    coords = np.vstack([cols, rows])

    # Extract the pixel intensities along the line
    profile = map_coordinates(image, coords)
    return profile


def get_profile_slice(img, slice_id, point1, point2, num_points):
    img_slice = img[:, :, slice_id]

    # Create 1D profiles
    img_profile = create_1d_profile(img_slice, point1, point2, num_points)

    return img_profile
