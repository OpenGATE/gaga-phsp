#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gaga_phsp as gaga
import garf
from garf.helpers import get_gpu_device
from scipy.spatial.transform import Rotation
import itk
import opengate.sources.gansources as gansources
from tqdm import tqdm
import torch
import opengate as gate


def voxelized_source_generator(source_filename):
    gen = gansources.VoxelizedSourceConditionGenerator(
        source_filename, use_activity_origin=True
    )
    gen.compute_directions = True
    return gen.generate_condition


def gaga_garf_load_nets_and_initialize(gaga_user_info, garf_user_info):
    # ensure int
    gaga_user_info.batch_size = int(gaga_user_info.batch_size)
    garf_user_info.batch_size = int(garf_user_info.batch_size)

    # load gaga pth
    gaga_params, G, D, _ = gaga.load(gaga_user_info.pth_filename)
    gaga_user_info.gaga_params = gaga_params
    gaga_user_info.G = G
    gaga_user_info.D = D

    # load garf pth
    nn, model = garf.load_nn(garf_user_info.pth_filename, verbose=False)
    garf_user_info.nn = nn
    garf_user_info.model_data = model

    # set gpu/cpu for garf
    current_gpu_mode, current_gpu_device = get_gpu_device(garf_user_info.gpu_mode)
    garf_user_info.nn.model_data["current_gpu_mode"] = current_gpu_mode
    garf_user_info.nn.model_data["current_gpu_device"] = current_gpu_device

    # garf image plane rotation
    # FIXME -> to change for GE NM 670
    # print("WARNING WARNING WARNING gaga_helpers_spect.py plane rotation ?")
    # r = Rotation.from_euler("x", 0, degrees=True)
    # garf_user_info.plane_rotation = r

    # image plane size
    garf_user_info.nb_energy_windows = garf_user_info.nn.model_data["n_ene_win"]
    size = garf_user_info.image_size
    spacing = garf_user_info.image_spacing
    garf_user_info.image_plane_size_mm = np.array(
        [size[0] * spacing[0], size[1] * spacing[1]]
    )

    # size and spacing must be np
    garf_user_info.image_spacing = np.array(garf_user_info.image_spacing)
    garf_user_info.image_hspacing = garf_user_info.image_spacing / 2.0
    garf_user_info.image_plane_hsize_mm = garf_user_info.image_plane_size_mm / 2


def do_nothing(a):
    pass


def gaga_garf_verbose_user_info(gaga_user_info, garf_user_info, gantry_rotations):
    print(f"GAGA pth                 = {gaga_user_info.pth_filename}")
    print(f"GARF pth                 = {garf_user_info.pth_filename}")
    print(f"GARF hits slice          = {garf_user_info.hit_slice}")
    print(f"Activity source          = {gaga_user_info.activity_source}")
    print(f"Number of energy windows = {garf_user_info.nb_energy_windows}")
    print(f"Image plane size (pixel) = {garf_user_info.image_size}")
    print(f"Image plane spacing (mm) = {garf_user_info.image_spacing}")
    print(f"Image plane size (mm)    = {garf_user_info.image_plane_size_mm}")
    print(f"Number of angles         = {len(gantry_rotations)}")
    print(f"GAGA batch size          = {gaga_user_info.batch_size:.1e}")
    print(f"GARF batch size          = {garf_user_info.batch_size:.1e}")
    print(
        f"GAGA GPU mode            = {gaga_user_info.gaga_params['current_gpu_mode']}"
    )
    print(
        f"GARF GPU mode            = {garf_user_info.nn.model_data['current_gpu_mode']}"
    )


def gaga_garf_generate_spect_OLD(
        gaga_user_info, garf_user_info, n, gantry_rotations, verbose=True
):
    # n must be int
    n = int(n)

    # allocate the initial list of images :
    # number of angles x number of energy windows x image size
    nbe = garf_user_info.nb_energy_windows
    size = garf_user_info.image_size
    spacing = garf_user_info.image_spacing
    data_size = [len(gantry_rotations), nbe, size[0], size[1]]
    data_img = np.zeros(data_size, dtype=np.float64)

    # verbose
    if verbose:
        gaga_garf_verbose_user_info(gaga_user_info, garf_user_info, gantry_rotations)

    # create the planes for each angle (with max number of values = batch_size)
    planes = []
    gaga_user_info.batch_size = int(gaga_user_info.batch_size)
    projected_points = [None] * len(gantry_rotations)
    for rot in gantry_rotations:
        plane = garf.arf_plane_init(garf_user_info, rot, gaga_user_info.batch_size)
        planes.append(plane)
        print('plane', plane)

    # initialize the condition generator
    f = gaga_user_info.activity_source
    cond_generator = gansources.VoxelizedSourceConditionGenerator(
        f, use_activity_origin=False  # FIXME true or false ?
    )
    cond_generator.compute_directions = True
    cond_generator.translation = gaga_user_info.cond_translation
    print(f'{cond_generator.translation=}')

    # prepare verbose
    verb_gaga_1 = do_nothing
    verb_gaga_2 = do_nothing
    verb_garf_1 = do_nothing
    if gaga_user_info.verbose > 0:
        verb_gaga_1 = tqdm.write
    if gaga_user_info.verbose > 1:
        verb_gaga_2 = tqdm.write
    if garf_user_info.verbose > 0:
        verb_garf_1 = tqdm.write

    # loop on GAGA batches
    current_n = 0
    pbar = tqdm(total=n)
    nb_hits_on_plane = [0] * len(gantry_rotations)
    nb_detected_hits = [0] * len(gantry_rotations)
    while current_n < n:
        # check generation of the exact nb of samples
        current_batch_size = gaga_user_info.batch_size
        if current_batch_size > n - current_n:
            current_batch_size = n - current_n
        verb_gaga_1(f"Current event = {current_n}/{n}")

        # generate samples
        x = gaga.generate_samples_with_vox_condition(
            gaga_user_info, cond_generator, current_batch_size
        )

        # FIXME filter Energy too low (?)

        # generate projections
        for i in range(len(gantry_rotations)):
            # project on plane
            plane = planes[i]
            px = garf.arf_plane_project(x, plane, garf_user_info.image_plane_size_mm)
            verb_gaga_2(f"\tAngle {i}, number of gamma reaching the plane = {len(px)}")
            nb_hits_on_plane[i] += len(px)
            if len(px) == 0:
                continue

            # Store projected points until garf_batch_size is full before build image
            cpx = projected_points[i]
            if cpx is None:
                if len(px) > garf_user_info.batch_size:
                    print(
                        f"Cannot use GARF, {len(px)} points while batch size is {garf_user_info.batch_size}"
                    )
                    exit(-1)
                projected_points[i] = px
            else:
                if len(cpx) + len(px) > garf_user_info.batch_size:
                    # build image
                    image = data_img[i]
                    verb_garf_1(
                        f"\tGARF rotation {i}: update image with {len(cpx)} hits ({current_n}/{n})"
                    )
                    nb_detected_hits[i] += garf.arf_build_image_from_projected_points_numpy(
                        garf_user_info, cpx, image
                    )
                    projected_points[i] = px
                else:
                    projected_points[i] = np.concatenate((cpx, px), axis=0)

            # next angles index
            i += 1

        # iterate
        current_n += current_batch_size
        pbar.update(current_batch_size)

    # remaining projected points
    for i in range(len(gantry_rotations)):
        cpx = projected_points[i]
        if cpx is None or len(cpx) == 0:
            continue
        if garf_user_info.verbose > 0:
            print(f"GARF rotation {i}: update image with {len(cpx)} hits (final)")
        image = data_img[i]
        nb_detected_hits[i] = garf.arf_build_image_from_projected_points_numpy(
            garf_user_info, cpx, image
        )

    if verbose:
        for i in range(len(gantry_rotations)):
            print(f"Angle {i}, nb of hits on plane = {nb_hits_on_plane[i]}")
            print(f"Angle {i}, nb of detected hits = {nb_detected_hits[i]}")

    # Remove first slice (nb of hits)
    if not garf_user_info.hit_slice:
        data_img = data_img[:, 1:, :]

    # Final list of images
    images = []
    for i in range(len(gantry_rotations)):
        img = itk.image_from_array(data_img[i])
        spacing = [spacing[0], spacing[1], 1]
        origin = [
            -size[0] * spacing[0] / 2 + spacing[0] / 2,
            -size[1] * spacing[1] / 2 + spacing[1] / 2,
            0,
        ]
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        images.append(img)
        i += 1

    return images


def generate_samples_with_vox_condition(gaga_user_info, cond_generator, n):
    # generate conditions
    cond = cond_generator.generate_condition(n)

    # generate samples
    x = gaga.generate_samples3(
        gaga_user_info.gaga_params,
        gaga_user_info.G,
        n=n,
        cond=cond,
    )

    # move backward
    pos_index = 1  # FIXME
    dir_index = 4
    position = x[:, pos_index: pos_index + 3]
    direction = x[:, dir_index: dir_index + 3]
    x[:, pos_index: pos_index + 3] = (
            position - gaga_user_info.backward_distance * direction
    )

    return x


def gaga_garf_generate_spect_torch_OLD_TO_REMOVE(
        gaga_user_info, garf_user_info, n, gantry_rotations, verbose=True
):
    # n must be int
    n = int(n)

    # allocate the initial list of images :
    # number of angles x number of energy windows x image size
    # nbe = garf_user_info.nb_energy_windows
    size = garf_user_info.image_size
    spacing = garf_user_info.image_spacing
    # data_size = [len(angle_rotations), nbe, size[0], size[1]]
    # data_img = np.zeros(data_size, dtype=np.float64)

    # verbose
    if verbose:
        gaga_garf_verbose_user_info(gaga_user_info, garf_user_info, gantry_rotations)

    # print
    print()
    print('-------------------------------')
    for k, v in zip(gaga_user_info.keys(), gaga_user_info.values()):
        if k != 'gaga_params':
            print(f"{k} = {v}")
    print()
    print('-------------------------------')
    for k, v in zip(garf_user_info.keys(), garf_user_info.values()):
        if k != 'nn':
            print(f"{k} = {v}")

    # device gpu
    current_gpu_mode, current_gpu_device = get_gpu_device(gaga_user_info.gpu_mode)

    # create the angle planes
    l_detectorsPlanes = []
    image_size = garf_user_info.image_size[0] * spacing[0]  # FIXME set as 2D
    print(f'image_size = {image_size}')
    print(f'angle = {gantry_rotations}')
    for angle in gantry_rotations:
        # FIXME (center)
        det_plane = DetectorPlane(size=image_size,  # FIXME set as 2D
                                  device=current_gpu_device,
                                  center0=[0, 0, -garf_user_info.plane_distance],  ## FIXME radius is SID ?
                                  # center0=[0, -garf_user_info.plane_distance, 0],  ## FIXME radius is SID ?
                                  rot_angle=angle,
                                  dist_to_crystal=garf_user_info.distance_to_crystal)
        l_detectorsPlanes.append(det_plane)
        print(det_plane)

    # main cond gan object
    cgan_source = CondGANSource(gaga_user_info.batch_size,
                                gaga_user_info.pth_filename,
                                current_gpu_device)

    # main garf object
    garf_ui = {}
    garf_ui['pth_filename'] = garf_user_info.pth_filename
    garf_ui['batchsize'] = garf_user_info.batch_size
    garf_ui['device'] = current_gpu_device
    garf_ui['output_fn'] = 'output/proj.mhd'
    garf_ui['nprojs'] = len(l_detectorsPlanes)
    garf_detector = GARF(user_info=garf_ui)

    # condition dataset
    dataset = ConditionsDataset(activity=n,
                                cgan_src=cgan_source,
                                source_fn=gaga_user_info.activity_source,
                                device=current_gpu_device,
                                save_cond=False)

    # loop on batch
    batch_size = int(gaga_user_info.batch_size)
    n_batchs = int(n // batch_size)
    print()
    print(f'n = {n}')
    print(f'batch_size = {batch_size}')
    print(f'n_batchs = {n_batchs}')
    N = 0
    M = 0
    pbar = tqdm(total=n)
    with torch.no_grad():
        for _ in range(n_batchs):
            gan_input_z_cond = dataset.get_batch(batch_size)
            N += batch_size
            print(f'Generate {batch_size} {N}')
            gan_input_z_cond = gan_input_z_cond.to(current_gpu_device)
            fake = cgan_source.generate(gan_input_z_cond)

            # FIXME: what is it ???
            fake = fake[fake[:, 0] > 0.100]
            dirn = torch.sqrt(fake[:, 4] ** 2 + fake[:, 5] ** 2 + fake[:, 6] ** 2)
            fake[:, 4:7] = fake[:, 4:7] / dirn[:, None]

            # FIXME ????
            # backproject a little bit: p2= p1 - alpha * d1
            # solved (avec inconnu=alpha) using ||p2||² = R2² puis equation degré 2 en alpha
            # beta=(fake[:,1:4] * fake[:,4:7]).sum(dim=1)
            # R1,R2 = 610,args.sid - 150
            # alpha= beta - torch.sqrt(beta**2 + R2**2-R1**2)
            # fake[:,1:4] = fake[:,1:4] - alpha[:,None]*fake[:,4:7]
            fake[:, 1:4] = fake[:, 1:4] - 600 * fake[:, 4:7]  # backward

            l_nc = []

            for proj_i, plane_i in enumerate(l_detectorsPlanes):
                batch_arf_i = plane_i.get_intersection(batch=fake)
                garf_detector.apply(batch_arf_i, proj_i)

            pbar.update(batch_size)

    # dataset.save_conditions("toto.mhd")

    garf_detector.save_projections()


def get_rot_matrix(theta):
    # use this if the desired rotation axis is "y" (default)
    print(f'theta = {theta}')
    print(f'theta en deg = {theta * 360 / 3.14}')
    theta = torch.tensor([theta])
    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                         [0, 1, 0],
                         [-torch.sin(theta), 0, torch.cos(theta)]])


def get_rot_matrix_x(theta):
    # use this if the desired rotation axis is "x"
    theta = torch.tensor([theta])
    return torch.tensor([[1, 0, 0],
                         [0, torch.cos(theta), torch.sin(theta)],
                         [0, -torch.sin(theta), torch.cos(theta)]])


def get_rot_matrix_z(theta):
    # use this if the desired rotation axis is "z"
    theta = torch.tensor([theta])
    return torch.tensor([[torch.cos(theta), torch.sin(theta), 0],
                         [-torch.sin(theta), torch.cos(theta), 0],
                         [0, 0, 1]]
                        )


def get_rot_matrix_test(theta):
    print(f'theta: {theta}')
    r1 = Rotation.from_euler("yx", (90, 90), degrees=True)
    # r2 = rotation X180 is to set the detector head-foot, rotation Z90 is the gantry angle
    r2 = Rotation.from_euler("xz", (180, 90), degrees=True)
    plane_rotation = r2 * r1
    rot = Rotation.from_euler("z", theta, degrees=True)
    final_rot = rot * plane_rotation
    m = final_rot.as_matrix().astype(np.float32)
    print(f'final_rot: {m}')
    # use this if the desired rotation axis is "z"
    theta = torch.tensor([theta])
    print(f'cos theta = {torch.cos(theta)}')
    '''return torch.tensor([
        [m[0, 0], m[0, 1], m[0, 2]],
        [m[1, 0], m[1, 1], m[1, 2]],
        [m[2, 0], m[2, 1], m[2, 2]]])'''
    return torch.tensor([
        [m[0, 0], m[1, 0], m[2, 0]],
        [m[0, 1], m[1, 1], m[2, 1]],
        [m[0, 2], m[1, 2], m[2, 2]]])


class VoxelizerSourcePDFSamplerTorch_v2:
    """
    This is an alternative to GateSPSVoxelsPosDistribution (c++)
    It is needed because the cond voxel source is used on python side.

    There are two versions, version 2 is much slower (do not use)
    """

    def __init__(self, itk_image, device, version=1):
        self.image = itk_image
        self.version = version
        # get image in np array
        self.imga = itk.array_view_from_image(itk_image)
        imga = self.imga

        # image sizes
        lx = self.imga.shape[0]
        ly = self.imga.shape[1]
        lz = self.imga.shape[2]

        # normalized pdf
        pdf = imga.ravel(order="F")
        self.pdf = pdf / pdf.sum()

        self.device = device

        self.pdf = torch.from_numpy(self.pdf).to()

        # create grid of indices
        [x_grid, y_grid, z_grid] = torch.meshgrid(
            torch.arange(lx).to(self.device), torch.arange(ly).to(self.device), torch.arange(lz).to(self.device),
            indexing="ij"
        )

        # list of indices
        self.xi, self.yi, self.zi = (
            x_grid.permute(2, 1, 0).contiguous().view(-1),
            y_grid.permute(2, 1, 0).contiguous().view(-1),
            z_grid.permute(2, 1, 0).contiguous().view(-1),
        )

    def sample_indices(self, n):
        indices = torch.multinomial(self.pdf, num_samples=n, replacement=True)
        i = self.xi[indices]
        j = self.yi[indices]
        k = self.zi[indices]
        return i, j, k


class GagaSource_OLDTOREMOVE:
    """
    FIXME : to replace generate_samples3
    """

    def __init__(self):
        # user input
        mm = gate.g4_units.mm
        self.gpu_mode = "auto"
        self.pth_filename = None
        self.activity_filename = None
        self.batch_size = 1e5
        self.backward_distance = 400 * mm

        self.cond_translation = None  # FIXME FIXME FIXME
        self.is_initialized = False

        # other members
        self.current_gpu_mode = None
        self.current_gpu_device = None
        self.gaga_params = None
        self.G = None
        self.z_rand = None
        self.x_mean = None
        self.x_std = None
        self.x_mean_cond = None
        self.x_std_cond = None
        self.x_mean_non_cond = None
        self.x_std_non_cond = None
        self.x_dim = None
        self.z_dim = None
        self.nb_cond_keys = None
        self.cond_activity = None

    def initialize(self):
        # gpu mode
        self.current_gpu_mode, self.current_gpu_device = get_gpu_device(self.gpu_mode)

        # int
        self.batch_size = int(self.batch_size)

        # load the GAN Generator
        self.gaga_params, self.G, _, __ = gaga.load(self.pth_filename, self.gpu_mode)
        self.G = self.G.to(self.current_gpu_device)
        self.G.eval()

        # initialize the z rand
        self.z_rand = gaga.init_z_rand(self.gaga_params)

        # initialize the mean/std
        self.initialize_normalization()

        # initialize the conditional voxelized source
        self.cond_activity = GagaConditionalVoxelizedActivity(self)

    def initialize_normalization(self):
        params = self.gaga_params
        # normalize the conditional vector
        self.x_mean = params["x_mean"][0]
        self.x_std = params["x_std"][0]
        xn = params["x_dim"]
        cn = len(params["cond_keys"])

        # needed ?
        self.nb_cond_keys = cn
        self.x_dim = xn
        self.z_dim = params["z_dim"]

        # which device ?
        dev = self.current_gpu_device
        if self.current_gpu_mode == "mps":
            self.x_mean = torch.tensor(self.x_mean.astype(np.float32), device=dev)
            self.x_std = torch.tensor(self.x_std.astype(np.float32), device=dev)
        else:
            self.x_mean = torch.tensor(self.x_mean, device=dev)
            self.x_std = torch.tensor(self.x_std, device=dev)

        # mean and std for cond only
        self.x_mean_cond = self.x_mean[xn - cn: xn]
        self.x_std_cond = self.x_std[xn - cn: xn]

        # mean and std for non cond
        self.x_mean_non_cond = self.x_mean[0: xn - cn]
        self.x_std_non_cond = self.x_std[0: xn - cn]

    def __str__(self):
        mm = gate.g4_units.mm
        s = f"gaga user gpu mode: {self.gpu_mode}\n"
        s += f"gaga current gpu mode: {self.current_gpu_mode}\n"
        s += f"gaga batch size: {self.batch_size}\n"
        s += f"gaga backward: {self.backward_distance / mm} mm\n"
        s += f"gaga nb conditions: {self.nb_cond_keys}\n"
        s += f"gaga x dim: {self.x_dim}\n"
        return s

    def generate_projections_numpy(self, garf_detector, n):
        print()

    def generate_projections_torch(self, garf_detector, n):
        pbar = tqdm(total=n)
        current_n = 0
        with torch.no_grad():
            while current_n < n:
                current_batch_size = min(self.batch_size, n - current_n)
                print(f"current n: {current_n}/{n}\n")
                print(f"current batch size: {current_batch_size}\n")
                fake = self.generate_particles(current_batch_size)
                current_n += current_batch_size

                # FIXME: what is it ??? E threshold ?
                # fake = fake[fake[:, 0] > 0.100]
                fake = fake[fake[:, 0] > 0.01]

                # FIXME normalize direction ?
                dirn = torch.sqrt(fake[:, 4] ** 2 + fake[:, 5] ** 2 + fake[:, 6] ** 2)
                fake[:, 4:7] = fake[:, 4:7] / dirn[:, None]

                # FIXME ????
                # backproject a little bit: p2= p1 - alpha * d1
                # solved (avec inconnu=alpha) using ||p2||² = R2² puis equation degré 2 en alpha
                # beta=(fake[:,1:4] * fake[:,4:7]).sum(dim=1)
                # R1,R2 = 610,args.sid - 150
                # alpha= beta - torch.sqrt(beta**2 + R2**2-R1**2)
                # fake[:,1:4] = fake[:,1:4] - alpha[:,None]*fake[:,4:7]
                # FIXME 600 is backward
                fake[:, 1:4] = fake[:, 1:4] - 600 * fake[:, 4:7]  # backward

                garf_detector.project_to_planes_torch(fake)
                """l_nc = []
                for proj_i, plane_i in enumerate(l_detectorsPlanes):
                    batch_arf_i = plane_i.get_intersection(batch=fake)
                    garf_detector.apply(batch_arf_i, proj_i)"""

                pbar.update(current_batch_size)

        images = garf_detector.save_projections()  # FIXME <---- build images, from gpu to cpu + offset
        print(images)
        return images

    def generate_projections_OLDTOREMOVE(self, garf_detector, n):
        pbar = tqdm(total=n)
        current_n = 0
        with torch.no_grad():
            while current_n < n:
                current_batch_size = min(self.batch_size, n - current_n)
                print(f"current n: {current_n}/{n}\n")
                print(f"current batch size: {current_batch_size}\n")
                fake = self.generate_particles(current_batch_size)
                current_n += current_batch_size

                # FIXME: what is it ??? E threshold ?
                # fake = fake[fake[:, 0] > 0.100]
                fake = fake[fake[:, 0] > 0.01]

                # FIXME normalize direction ?
                dirn = torch.sqrt(fake[:, 4] ** 2 + fake[:, 5] ** 2 + fake[:, 6] ** 2)
                fake[:, 4:7] = fake[:, 4:7] / dirn[:, None]

                # FIXME ????
                # backproject a little bit: p2= p1 - alpha * d1
                # solved (avec inconnu=alpha) using ||p2||² = R2² puis equation degré 2 en alpha
                # beta=(fake[:,1:4] * fake[:,4:7]).sum(dim=1)
                # R1,R2 = 610,args.sid - 150
                # alpha= beta - torch.sqrt(beta**2 + R2**2-R1**2)
                # fake[:,1:4] = fake[:,1:4] - alpha[:,None]*fake[:,4:7]
                # FIXME 600 is backward
                fake[:, 1:4] = fake[:, 1:4] - 600 * fake[:, 4:7]  # backward

                garf_detector.project_to_planes_torch(fake)
                """l_nc = []
                for proj_i, plane_i in enumerate(l_detectorsPlanes):
                    batch_arf_i = plane_i.get_intersection(batch=fake)
                    garf_detector.apply(batch_arf_i, proj_i)"""

                pbar.update(current_batch_size)

        images = garf_detector.save_projections()  # FIXME <---- build images, from gpu to cpu + offset
        print(images)
        return images

    def generate_particles(self, n):
        # get the conditions
        vox_cond = self.cond_activity.generate_conditions(n)
        vox_cond = vox_cond.to(self.current_gpu_device)

        # go !
        fake = self.G(vox_cond)

        # un-normalize
        fake = (fake * self.x_std_non_cond) + self.x_mean_non_cond

        return fake.float()


class GagaSource:
    """
    FIXME : to replace generate_samples3
    """

    def __init__(self):
        mm = gate.g4_units.mm
        # user input
        self.gpu_mode = "auto"
        self.pth_filename = None
        self.activity_filename = None
        self.batch_size = 1e5
        self.backward_distance = 400 * mm

        self.cond_translation = None  # FIXME FIXME FIXME

        # other members
        self.is_initialized = False
        self.current_gpu_mode = None
        self.current_gpu_device = None
        self.gaga_params = None
        self.G = None
        self.z_rand = None
        self.x_mean = None
        self.x_std = None
        self.x_mean_cond = None
        self.x_std_cond = None
        self.x_mean_non_cond = None
        self.x_std_non_cond = None
        self.x_dim = None
        self.z_dim = None
        self.nb_cond_keys = None
        self.cond_activity = None

    def __str__(self):
        mm = gate.g4_units.mm
        s = f"gaga user gpu mode: {self.gpu_mode}\n"
        s += f"gaga current gpu mode: {self.current_gpu_mode}\n"
        s += f"gaga pth_filename: {self.pth_filename}\n"
        s += f"gaga batch size: {self.batch_size}\n"
        s += f"gaga backward_distance: {self.backward_distance / mm} mm\n"
        s += f"gaga nb conditions: {self.nb_cond_keys}\n"
        s += f"gaga x dim: {self.x_dim}\n"
        return s

    def initialize(self):
        print(f'GagaSource initialize')
        if self.is_initialized:
            raise Exception(f'GagaSource is already initialized')

        # gpu mode
        self.current_gpu_mode, self.current_gpu_device = get_gpu_device(self.gpu_mode)

        # int
        self.batch_size = int(self.batch_size)

        # load the GAN Generator
        self.gaga_params, self.G, _, __ = gaga.load(self.pth_filename, self.gpu_mode)
        self.G = self.G.to(self.current_gpu_device)
        self.G.eval()

        # initialize the z rand
        self.z_rand = gaga.init_z_rand(self.gaga_params)

        # initialize the mean/std
        self.initialize_normalization()

        # initialize the conditional voxelized source
        self.cond_activity = GagaConditionalVoxelizedActivity(self)
        print(f'{self.cond_activity=}')

        self.is_initialized = True

    def initialize_normalization(self):
        params = self.gaga_params
        # normalize the conditional vector
        self.x_mean = params["x_mean"][0]
        self.x_std = params["x_std"][0]
        xn = params["x_dim"]
        cn = len(params["cond_keys"])

        # needed ?
        self.nb_cond_keys = cn
        self.x_dim = xn
        self.z_dim = params["z_dim"]

        # which device ?
        dev = self.current_gpu_device
        if self.current_gpu_mode == "mps":
            self.x_mean = torch.tensor(self.x_mean.astype(np.float32), device=dev)
            self.x_std = torch.tensor(self.x_std.astype(np.float32), device=dev)
        else:
            self.x_mean = torch.tensor(self.x_mean, device=dev)
            self.x_std = torch.tensor(self.x_std, device=dev)

        # mean and std for cond only
        self.x_mean_cond = self.x_mean[xn - cn: xn]
        self.x_std_cond = self.x_std[xn - cn: xn]

        # mean and std for non cond
        self.x_mean_non_cond = self.x_mean[0: xn - cn]
        self.x_std_non_cond = self.x_std[0: xn - cn]

    def generate_projections_numpy(self, garf_detector, n):
        if self.is_initialized is False:
            raise Exception(f'GarDetector must be initialized')
        n = int(n)
        nb_angles = len(garf_detector.plane_rotations)

        # create the planes for each angle
        planes = garf_detector.initialize_planes_numpy(self.batch_size)
        projected_points = [None] * nb_angles

        # cond GAN generator
        cond_generator = gansources.VoxelizedSourceConditionGenerator(
            self.activity_filename,
            use_activity_origin=False  # FIXME true or false ?
        )
        cond_generator.compute_directions = True
        cond_generator.translation = self.cond_translation

        # STEP2.5 alloc
        nbe = garf_detector.nb_ene
        size = garf_detector.image_size
        spacing = garf_detector.image_spacing
        data_size = [nb_angles, nbe, int(size[0]), int(size[1])]
        data_img = np.zeros(data_size, dtype=np.float64)

        # loop on GAGA batches
        current_n = 0
        pbar = tqdm(total=n)
        while current_n < n:
            current_batch_size = self.batch_size
            if current_batch_size > n - current_n:
                current_batch_size = n - current_n

            # generate samples
            fake = self.generate_particles_numpy(cond_generator, current_batch_size)

            # generate projections
            for i in range(nb_angles):
                garf_detector.project_to_planes_numpy(fake, i, planes, projected_points, data_img)
                i += 1
            # iterate
            current_n += current_batch_size
            pbar.update(current_batch_size)

        # STEP4
        # remaining projected points
        for i in range(nb_angles):
            cpx = projected_points[i]
            if cpx is None or len(cpx) == 0:
                continue
            image = data_img[i]
            garf_detector.build_image_from_projected_points_numpy(cpx, image)

        # Remove first slice (nb of hits) # FIXME use a flag
        data_img = data_img[:, 1:, :]

        # Final list of images
        images = []
        for i in range(nb_angles):
            img = itk.image_from_array(data_img[i])
            spacing = np.array([spacing[0], spacing[1], 1.0])
            origin = [
                -size[0] * spacing[0] / 2 + spacing[0] / 2,
                -size[1] * spacing[1] / 2 + spacing[1] / 2,
                0,
            ]
            img.SetOrigin(origin)
            img.SetSpacing(spacing)
            images.append(img)
            i += 1

        return images

    def generate_projections_torch(self, garf_detector, n):
        if self.is_initialized is False:
            raise Exception(f'GagaSource must be initialized')
        # specific initialisation for torch
        garf_detector.initialize_torch()
        # start progress bar
        pbar = tqdm(total=n)
        # main loop
        current_n = 0
        with torch.no_grad():
            while current_n < n:
                current_batch_size = min(self.batch_size, n - current_n)
                print(f"current n: {current_n}/{n} (batch = {current_batch_size}")
                fake = self.generate_particles_torch(current_batch_size)
                current_n += current_batch_size

                # project on detector plane
                garf_detector.project_to_planes_torch(fake)

                # progress bar
                pbar.update(current_batch_size)

        images = garf_detector.save_projections()
        return images

    def generate_projections_OLDTOREMOVE(self, garf_detector, n):
        pbar = tqdm(total=n)
        current_n = 0
        with torch.no_grad():
            while current_n < n:
                current_batch_size = min(self.batch_size, n - current_n)
                print(f"current n: {current_n}/{n}\n")
                print(f"current batch size: {current_batch_size}\n")
                fake = self.generate_particles_torch(current_batch_size)
                current_n += current_batch_size

                # FIXME: what is it ??? E threshold ?
                # fake = fake[fake[:, 0] > 0.100]
                fake = fake[fake[:, 0] > 0.01]

                # FIXME normalize direction ?
                dirn = torch.sqrt(fake[:, 4] ** 2 + fake[:, 5] ** 2 + fake[:, 6] ** 2)
                fake[:, 4:7] = fake[:, 4:7] / dirn[:, None]

                # FIXME ????
                # backproject a little bit: p2= p1 - alpha * d1
                # solved (avec inconnu=alpha) using ||p2||² = R2² puis equation degré 2 en alpha
                # beta=(fake[:,1:4] * fake[:,4:7]).sum(dim=1)
                # R1,R2 = 610,args.sid - 150
                # alpha= beta - torch.sqrt(beta**2 + R2**2-R1**2)
                # fake[:,1:4] = fake[:,1:4] - alpha[:,None]*fake[:,4:7]
                # FIXME 600 is backward
                fake[:, 1:4] = fake[:, 1:4] - 600 * fake[:, 4:7]  # backward

                garf_detector.project_to_planes_torch(fake)
                """l_nc = []
                for proj_i, plane_i in enumerate(l_detectorsPlanes):
                    batch_arf_i = plane_i.get_intersection(batch=fake)
                    garf_detector.apply(batch_arf_i, proj_i)"""

                pbar.update(current_batch_size)

        images = garf_detector.save_projections()  # FIXME <---- build images, from gpu to cpu + offset
        print(images)
        return images

    def generate_particles_torch(self, n):
        # get the conditions
        vox_cond = self.cond_activity.generate_conditions(n)
        vox_cond = vox_cond.to(self.current_gpu_device)
        # go !
        fake = self.G(vox_cond)
        # un-normalize
        fake = (fake * self.x_std_non_cond) + self.x_mean_non_cond

        # FIXME: what is it ??? E threshold ?
        # fake = fake[fake[:, 0] > 0.100]
        fake = fake[fake[:, 0] > 0.01]

        # FIXME normalize direction ?
        dirn = torch.sqrt(fake[:, 4] ** 2 + fake[:, 5] ** 2 + fake[:, 6] ** 2)
        fake[:, 4:7] = fake[:, 4:7] / dirn[:, None]

        # move backward
        fake[:, 1:4] = fake[:, 1:4] - self.backward_distance * fake[:, 4:7]

        return fake.float()

    def generate_particles_numpy(self, cond_generator, n):
        # generate conditions
        cond = cond_generator.generate_condition(n)

        # generate samples
        x = gaga.generate_samples3(
            self.gaga_params,
            self.G,
            n=n,
            cond=cond,
        )

        # move backward
        pos_index = 1  # FIXME
        dir_index = 4
        position = x[:, pos_index: pos_index + 3]
        direction = x[:, dir_index: dir_index + 3]
        x[:, pos_index: pos_index + 3] = (
                position - self.backward_distance * direction
        )

        # FIXME filter Energy too low (?)

        return x


class GagaConditionalVoxelizedActivity:

    def __init__(self, gaga_source):
        print()
        self.gaga_source = gaga_source

        # members
        self.sampler = None
        self.translation = None

        # init
        self.initialize()

    def initialize(self):
        print(f'GagaConditionalVoxelizedActivity.initialize()')
        # read activity image
        source = itk.imread(self.gaga_source.activity_filename)
        source_array = itk.array_from_image(source)

        # device
        dev = self.gaga_source.current_gpu_device

        # compute the offset
        self.source_size = np.array(source_array.shape)
        self.source_spacing = np.array(source.GetSpacing())
        # self.source_origin = np.array(source.GetOrigin())
        self.translation = (self.source_size - 1) * self.source_spacing / 2
        if self.gaga_source.current_gpu_mode == 'mps':
            self.translation = self.translation.astype(np.float32)
        self.translation = torch.from_numpy(self.translation).to(dev).flip(0)
        print(f'translation = {self.translation}')

        # shorter name for current gpu device
        # self.device = self.gaga_source.current_gpu_device

        self.sampler = VoxelizerSourcePDFSamplerTorch_v2(source, dev)
        # self.xmeanc = cgan_src.x_mean_cond.to(self.device)
        # self.xstdc = cgan_src.x_std_cond.to(self.device)
        # self.z_dim = cgan_src.z_dim
        # self.z_rand = cgan_src.z_rand

    def generate_conditions(self, n):
        print(f'Generating conditions {n}')
        # sample the voxels
        i, j, k = self.sampler.sample_indices(n=n)

        # half pixel size
        hs = self.source_spacing / 2.0

        # device
        dev = self.gaga_source.current_gpu_device

        # sample within the voxels
        rx = torch.rand(n, device=dev) * 2 * hs[0] - hs[0]
        ry = torch.rand(n, device=dev) * 2 * hs[1] - hs[1]
        rz = torch.rand(n, device=dev) * 2 * hs[2] - hs[2]

        # warning order np is z,y,x while itk is x,y,z
        x = self.source_spacing[0] * i + rz
        y = self.source_spacing[1] * j + ry
        z = self.source_spacing[2] * k + rx
        # FIXME t ???
        t = torch.tensor([5.99999588, -28.00000624, -402.0000022], device=dev)
        p = torch.column_stack((z, y, x)) - self.translation + t
        print(f"{self.translation=}")

        # sample direction
        directions = self.generate_isotropic_directions_torch(n)
        cond_x = torch.column_stack((p, directions))

        # apply un-normalization (needed)
        xm = self.gaga_source.x_mean_cond
        xs = self.gaga_source.x_std_cond
        cond_x = (cond_x - xm) / xs

        # generate the random z
        z = self.gaga_source.z_rand((n, self.gaga_source.z_dim), device=dev)

        # concat conditions + z
        vox_cond = torch.cat((z, cond_x), dim=1).float()
        return vox_cond

    def generate_isotropic_directions_torch(self, n):
        # device
        dev = self.gaga_source.current_gpu_device

        # angles
        min_theta = torch.tensor([0], device=dev)
        max_theta = torch.tensor([torch.pi], device=dev)
        min_phi = torch.tensor([0], device=dev)
        max_phi = 2 * torch.tensor([torch.pi], device=dev)

        u = torch.rand(n, device=dev)
        cos_theta = torch.cos(min_theta) - u * (torch.cos(min_theta) - torch.cos(max_theta))
        sin_theta = torch.sqrt(1 - cos_theta ** 2)

        v = torch.rand(n, device=dev)
        phi = min_phi + (max_phi - min_phi) * v
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        px = -sin_theta * cos_phi
        py = -sin_theta * sin_phi
        pz = -cos_theta

        return torch.column_stack((px, py, pz))


def generate_spect_images_np(gaga_source, garf_detector):
    print()
    # init
    # verbose
    # create garf planes information
    # create gaga sources
    # loop on batches
    #      generate conditions
    #      generate photon sources
    #      loop on planes
    #           compute intersection
    #           apply garf


def generate_spect_images_torch_OLDTOREMOVE(gaga_source, garf_detector, center, rotation_matrices, n):
    # FIXME in gaga_source object ?
    print()
    # init
    gaga_source.initialize()  ## FIXME can be in generate_spect_images

    # create garf planes information

    # FIXME plane must be init before
    garf_detector.initialize1()  ## FIXME can be in generate_spect_images
    garf_detector.initialize_detector_plane_rotations(center, rotation_matrices)  ## FIXME can be in generate_spect_images
    garf_detector.initialize2()  ## FIXME can be in generate_spect_images

    # verbose
    print(gaga_source)
    print(garf_detector)

    # create gaga sources
    images = gaga_source.generate_projections_OLDTOREMOVE(garf_detector, n)
    # loop on batches
    #      generate conditions
    #      generate photon sources
    #      loop on planes
    #           compute intersection
    #           apply garf
    return images
