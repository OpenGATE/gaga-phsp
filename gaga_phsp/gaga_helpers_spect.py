#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gaga_phsp as gaga
import gaga_helpers
import garf
from garf.helpers import get_gpu_device
from scipy.spatial.transform import Rotation
import itk
import opengate.sources.gansources as gansources
from tqdm import tqdm
from box import Box
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
    r = Rotation.from_euler("x", 0, degrees=True)
    garf_user_info.plane_rotation = r

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


def gaga_garf_generate_spect(
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

    # initialize the condition generator
    f = gaga_user_info.activity_source
    cond_generator = gansources.VoxelizedSourceConditionGenerator(
        f, use_activity_origin=False  # FIXME true or false ?
    )
    cond_generator.compute_directions = True
    cond_generator.translation = gaga_user_info.cond_translation

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
                    nb_detected_hits[i] += garf.build_arf_image_from_projected_points(
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
        nb_detected_hits[i] = garf.build_arf_image_from_projected_points(
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


def gaga_garf_generate_spect_torch(
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

    garf_detector.save_projection()


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


class DetectorPlane:
    def __init__(self, size, device, center0, rot_angle, dist_to_crystal):
        print(f'rotation angle: {rot_angle}')
        self.device = device
        self.M = get_rot_matrix(rot_angle)
        self.Mt = get_rot_matrix(-rot_angle).to(device)
        self.center = torch.matmul(self.M, torch.tensor(center0).float()).to(device)
        self.normal = -self.center / torch.norm(self.center)
        self.dd = torch.matmul(self.center, self.normal)
        self.size = size
        self.dist_to_crystal = dist_to_crystal

    def get_intersection(self, batch):
        energ0 = batch[:, 0:1]
        pos0 = batch[:, 1:4]
        dir0 = batch[:, 4:7]

        dir_produit_scalaire = torch.sum(dir0 * self.normal, dim=1)
        t = (self.dd - torch.sum(pos0 * self.normal, dim=1)) / dir_produit_scalaire

        # position of the intersection
        pos_xyz = dir0 * t[:, None] + pos0

        pos_xyz_rot = torch.matmul(pos_xyz, self.Mt.t())
        dir_xyz_rot = torch.matmul(dir0, self.Mt.t())
        dir_xy_rot = dir_xyz_rot[:, 0:2]
        pos_xy_rot_crystal = pos_xyz_rot[:, 0:2] + self.dist_to_crystal * dir_xy_rot

        # pos_xy_rot = torch.matmul(pos_xyz, self.Mt[[0,2], :].t()) # use this instead if the desired rotation axis is "z"
        # dir_xy_rot = torch.matmul(dir0, self.Mt[[0,2], :].t()) # use this instead if the desired rotation axis is "z"

        # pos_xyz_rot_crystal = pos_xyz_rot + (self.dist_to_crystal/dir_xyz_rot[:,2:3]) * dir_xyz_rot
        # pos_xy_rot_crystal = pos_xyz_rot_crystal[:,0:2]

        indexes_to_keep = torch.where((dir_produit_scalaire < 0) &
                                      (t > 0) &
                                      (pos_xy_rot_crystal.abs().max(dim=1)[0] < self.size / 2)
                                      )[0]

        batch_arf = torch.concat((pos_xy_rot_crystal[indexes_to_keep, :],
                                  dir_xy_rot[indexes_to_keep, :],
                                  energ0[indexes_to_keep, :]), dim=1)

        return batch_arf


class ConditionsDataset:
    def __init__(self, activity, cgan_src, source_fn, device, save_cond=False):
        self.total_activity = int(float(activity))
        source = itk.imread(source_fn)

        source_array = itk.array_from_image(source)
        self.source_size = np.array(source_array.shape)
        self.source_spacing = np.array(source.GetSpacing())
        self.source_origin = np.array(source.GetOrigin())
        self.offset = (self.source_size - 1) * self.source_spacing / 2

        self.save_cond = save_cond

        if save_cond:
            self.condition_img = np.zeros(self.source_size)

        self.device = device

        self.sampler = VoxelizerSourcePDFSamplerTorch(source, device=device)
        self.xmeanc = cgan_src.xmeanc.to(self.device)
        self.xstdc = cgan_src.xstdc.to(self.device)
        self.z_dim = cgan_src.z_dim
        self.z_rand = cgan_src.z_rand
        # FIXME np.float32 for MPS only
        self.offset = torch.from_numpy(self.offset.astype(np.float32)).to(self.device).flip(0)
        print(self.offset)

    def save_conditions(self, fn):
        condition_img_itk = itk.image_from_array(self.condition_img)
        condition_img_itk.SetSpacing(self.source_spacing)
        condition_img_itk.SetOrigin(self.source_origin)
        itk.imwrite(condition_img_itk, fn)

    def generate_isotropic_directions_torch(self, n):
        min_theta = torch.tensor([0], device=self.device)
        max_theta = torch.tensor([torch.pi], device=self.device)
        min_phi = torch.tensor([0], device=self.device)
        max_phi = 2 * torch.tensor([torch.pi], device=self.device)

        u = torch.rand(n, device=self.device)
        costheta = torch.cos(min_theta) - u * (torch.cos(min_theta) - torch.cos(max_theta))
        sintheta = torch.sqrt(1 - costheta ** 2)

        v = torch.rand(n, device=self.device)
        phi = min_phi + (max_phi - min_phi) * v
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)

        px = -sintheta * cosphi
        py = -sintheta * sinphi
        pz = -costheta

        return torch.column_stack((px, py, pz))

    def get_batch(self, n):
        i, j, k = self.sampler.sample_indices(n=n)

        if self.save_cond:
            for ii, jj, kk in zip(i, j, k):
                id_i, id_j, id_k = ii.cpu().numpy(), jj.cpu().numpy(), kk.cpu().numpy()
                self.condition_img[id_i, id_j, id_k] += 1

        # half pixel size
        hs = self.source_spacing / 2.0
        # sample within the voxel
        rx = torch.rand(n, device=self.device) * 2 * hs[0] - hs[0]
        ry = torch.rand(n, device=self.device) * 2 * hs[1] - hs[1]
        rz = torch.rand(n, device=self.device) * 2 * hs[2] - hs[2]
        # warning order np is z,y,x while itk is x,y,z
        x = self.source_spacing[0] * i + rz
        y = self.source_spacing[1] * j + ry
        z = self.source_spacing[2] * k + rx

        print(f'offset = - {self.offset}')
        translation = torch.tensor([5.99999588, -28.00000624, -402.0000022],
                                   device=self.device)  # FIXME translation offset
        p = torch.column_stack((z, y, x)) - self.offset + translation
        dir = self.generate_isotropic_directions_torch(n)
        condx = torch.column_stack((p, dir))

        condx = (condx - self.xmeanc) / self.xstdc
        z = self.z_rand((n, self.z_dim), device=self.device)
        gan_input_z_cond = torch.cat((z, condx), dim=1).float()
        return gan_input_z_cond


class VoxelizerSourcePDFSamplerTorch:
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


class GARF:
    def __init__(self, user_info):
        self.batchsize = user_info['batchsize']
        self.pth_filename = user_info['pth_filename']
        self.output_fn = user_info['output_fn']
        self.device = user_info['device']
        self.nprojs = user_info['nprojs']

        self.size = 256
        self.spacing = 2.3976
        self.image_size = [2 * self.nprojs, 256, 256]
        self.image_spacing = [self.spacing, self.spacing, 1]

        self.zeros = torch.zeros((self.image_size[1], self.image_size[2])).to(self.device)

        self.degree = np.pi / 180
        self.init_garf()

    def init_garf(self):
        # load the pth file
        self.nn, self.model = garf.load_nn(
            self.pth_filename, verbose=False
        )
        self.model = self.model.to(self.device)

        # size and spacing (2D)
        self.model_data = self.nn["model_data"]

        # FIXME MPS .astype(np.float32)
        self.x_mean = torch.tensor(self.model_data['x_mean'].astype(np.float32), device=self.device)
        self.x_std = torch.tensor(self.model_data['x_std'].astype(np.float32), device=self.device)
        if ('rr' in self.model_data):
            self.rr = self.model_data['rr']
        else:
            self.rr = self.model_data['RR']

        # output image: nb of energy windows times nb of runs (for rotation)
        self.nb_ene = self.model_data["n_ene_win"]

        # create output image as tensor
        self.output_image = torch.zeros(tuple(self.image_size)).to(self.device)
        # compute offset
        self.psize = [self.size * self.spacing, self.size * self.spacing]

        self.hsize = np.divide(self.psize, 2.0)
        self.offset = [self.image_spacing[0] / 2.0, self.image_spacing[1] / 2.0]

        print('--------------------------------------------------')
        print(f'size {self.size}')
        print(f'spacing {self.spacing}')
        print(f'image size {self.image_size}')
        print(f'image spacing {self.image_spacing}')
        print(f'psize : {self.psize}')
        print(f'psize : {self.psize}')

    def apply(self, batch, proj_i):  # build_arf_image_from_projected_points
        x = batch.clone()

        x[:, 2] = torch.arccos(batch[:, 2]) / self.degree
        x[:, 3] = torch.arccos(batch[:, 3]) / self.degree
        ax = x[:, 2:5]  # two angles and energy

        w = self.nn_predict(self.model, self.nn["model_data"], ax)

        # w = torch.bernoulli(w)
        # w = w.multinomial(1,replacement=True)

        # positions
        cx = x[:, 0:2]
        coord = (cx + (self.size - 1) * self.spacing / 2) / self.spacing
        vu = torch.round(coord).to(int)

        # vu, w_pred = self.remove_out_of_image_boundaries(vu, w, self.image_size)

        # do nothing if there is no hit in the image
        if vu.shape[0] != 0:
            # PW
            temp = self.zeros.fill_(0)
            temp = self.image_from_coordinates(temp, vu, w[:, 3])  # FIXME energy windows index !!
            self.output_image[proj_i, :, :] = self.output_image[proj_i, :, :] + temp
            # SW
            temp = self.zeros.fill_(0)
            temp = self.image_from_coordinates(temp, vu, w[:, 2])  # FIXME energy windows index !!
            self.output_image[proj_i + self.nprojs, :, :] = self.output_image[proj_i + self.nprojs, :, :] + temp

    def nn_predict(self, model, model_data, x):  # FIXME in garf_apply.py nn_predict_torch
        '''
        Apply the NN to predict y from x
        '''

        # apply input model normalisation
        x = (x - self.x_mean) / self.x_std

        vx = x.float()

        # predict values
        vy_pred = model(vx)

        # normalize probabilities
        y_pred = vy_pred
        y_pred = self.normalize_logproba(y_pred)
        y_pred = self.normalize_proba_with_russian_roulette(y_pred, 0, self.rr)

        return y_pred

    def normalize_logproba(self, x):
        '''
        Convert un-normalized log probabilities to normalized ones (0-100%)
        Not clear how to deal with exp overflow ?
        (https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)
        '''

        # exb = torch.exp(x)
        # exb_sum = torch.sum(exb, dim=1)
        # # divide if not equal at zero
        # p = torch.divide(exb.T, exb_sum,
        #               out=torch.zeros_like(exb.T)).T

        # check (should be equal to 1.0)
        # check = np.sum(p, axis=1)
        # print(check)

        b = x.amax(dim=1, keepdim=True)
        exb = torch.exp(x - b)
        exb_sum = torch.sum(exb, dim=1)
        p = torch.divide(exb.T, exb_sum, out=torch.zeros_like(exb.T)).T

        return p

    def compute_angle_offset(self, angles, length):
        '''
        compute the x,y offset according to the angle
        '''

        angles_rad = (angles) * np.pi / 180
        cos_theta = torch.cos(angles_rad[:, 0])
        cos_phi = torch.cos(angles_rad[:, 1])

        tx = length * cos_phi  ## yes see in Gate_NN_ARF_Actor, line "phi = acos(dir.x())/degree;"
        ty = length * cos_theta  ## yes see in Gate_NN_ARF_Actor, line "theta = acos(dir.y())/degree;"
        t = torch.column_stack((tx, ty))

        return t

    def image_from_coordinates(self, img, vu, w):
        img_r = img.ravel()
        ind_r = vu[:, 1] * img.shape[0] + vu[:, 0]
        img_r.put_(index=ind_r, source=w, accumulate=True)
        img = img_r.reshape_as(img)
        return img

    def remove_out_of_image_boundaries(self, vu, w_pred, size):
        '''
        Remove values out of the images (<0 or > size)
        '''
        len_0 = vu.shape[0]
        # index = torch.where((vu[:,0]>=0)
        #                     & (vu[:,1]>=0)
        #                     & (vu[:,0]< size[2])
        #                     & (vu[:,1]<size[1]))[0]
        # vu = vu[index]
        # w_pred = w_pred[index]

        vu_ = vu[(vu[:, 0] >= 0) & (vu[:, 1] >= 0) & (vu[:, 0] < size[2]) & (vu[:, 1] < size[1])]
        w_pred_ = w_pred[(vu[:, 0] >= 0) & (vu[:, 1] >= 0) & (vu[:, 0] < size[2]) & (vu[:, 1] < size[1])]

        if (len_0 - vu.shape[0] > 0):
            print('Remove points out of the image: {} values removed sur {}'.format(len_0 - vu.shape[0], len_0))

        return vu_, w_pred_

    # -----------------------------------------------------------------------------
    def normalize_proba_with_russian_roulette(self, w_pred, channel, rr):
        '''
        Consider rr times the values for the energy windows channel
        '''
        # multiply column 'channel' by rr
        w_pred[:, channel] *= rr
        # normalize
        p_sum = torch.sum(w_pred, dim=1, keepdim=True)
        w_pred = w_pred / p_sum
        # check
        # p_sum = np.sum(w_pred, axis=1)
        # print(p_sum)
        return w_pred

    def save_projection(self):
        # convert to itk image
        self.output_projections_array = self.output_image.cpu().numpy()
        self.output_projections_itk = itk.image_from_array(self.output_projections_array)

        # set spacing and origin like DigitizerProjectionActor
        spacing = self.image_spacing
        spacing = np.array([spacing[0], spacing[1], 1])
        size = np.array(self.image_size)
        size[0] = self.image_size[2]
        size[2] = self.image_size[0]
        origin = -size / 2.0 * spacing + spacing / 2.0
        self.output_projections_itk.SetSpacing(spacing)
        self.output_projections_itk.SetOrigin(origin)

        itk.imwrite(self.output_projections_itk, self.output_fn)
        print(f'Output projection saved in : {self.output_fn}')

        # SC
        k = 0.5
        self.output_projections_SC_array = self.output_projections_array[:self.nprojs, :,
                                           :] - k * self.output_projections_array[self.nprojs:, :, :]
        self.output_projections_SC_array[self.output_projections_SC_array < 0] = 0
        self.output_projections_SC_itk = itk.image_from_array(self.output_projections_SC_array)
        size = np.array([256, 256, self.nprojs])
        origin = -size / 2.0 * spacing + spacing / 2.0
        self.output_projections_SC_itk.SetSpacing(spacing)
        self.output_projections_SC_itk.SetOrigin(origin)
        projs_SC_fn = self.output_fn.replace('.mhd', '_SC.mhd')
        itk.imwrite(self.output_projections_SC_itk, projs_SC_fn)
        print(f'Output projection (SC) saved in : {projs_SC_fn}')


class CondGANSource:
    def __init__(self, batch_size, pth_filename, device):
        self.batchsize = int(float(batch_size))
        self.pth_filename = pth_filename
        self.device = device
        self.init_gan()
        self.condition_time = 0
        self.gan_time = 0

    def init_gan(self):
        self.gan_info = Box()
        g = self.gan_info
        g.params, g.G, _, __ = gaga.load(
            self.pth_filename, "auto"
        )
        g.G = g.G.to(self.device)
        g.G.eval()

        self.z_rand = self.get_z_rand(g.params)

        # normalize the conditional vector
        print(f"gpu mode = {g.params.current_gpu_mode}")
        if g.params.current_gpu_mode == "mps":
            xmean = torch.tensor(g.params["x_mean"][0].astype(np.float32), device=self.device)
            xstd = torch.tensor(g.params["x_std"][0].astype(np.float32), device=self.device)
        else:
            xmean = torch.tensor(g.params["x_mean"][0], device=self.device)
            xstd = torch.tensor(g.params["x_std"][0], device=self.device)
        xn = g.params["x_dim"]
        cn = len(g.params["cond_keys"])
        self.ncond = cn
        # mean and std for cond only
        self.xmeanc = xmean[xn - cn: xn]
        self.xstdc = xstd[xn - cn: xn]
        # mean and std for non cond
        self.xmeannc = xmean[0: xn - cn]
        self.xstdnc = xstd[0: xn - cn]
        print(f"mean nc : {self.xmeannc}")
        print(f"mean c : {self.xmeanc}")
        print(f"std nc : {self.xstdnc}")
        print(f"std c : {self.xstdc}")

        self.z_dim = g.params["z_dim"]
        self.x_dim = g.params["x_dim"]

        print(f"zdim : {self.z_dim}")
        print(f"xdim : {self.x_dim}")

    def get_z_rand(self, params):
        if "z_rand_type" in params:
            if params["z_rand_type"] == "rand":
                return torch.rand
            if params["z_rand_type"] == "randn":
                return torch.randn
        if "z_rand" in params:
            if params["z_rand"] == "uniform":
                return torch.rand
            if params["z_rand"] == "normal":
                return torch.randn
        params["z_rand_type"] = "randn"
        return torch.randn

    def generate(self, z):
        fake = self.gan_info.G(z)
        fake = (fake * self.xstdnc) + self.xmeannc
        return fake.float()


class GagaSource:
    """
    FIXME : to replace generate_samples3
    """

    def __init__(self):
        # user input
        mm = gate.g4_units.mm
        self.gpu_mode = "auto"
        self.pth_filename = None
        self.batch_size = 1e5
        self.backward_distance = 400 * mm

        # other members
        self.current_gpu_mode = None
        self.current_gpu_device = None
        self.gaga_params = None
        self.G = None
        self.z_rand = None
        self.xmean = None
        self.xstd = None
        self.xmeanc = None
        self.xstdc = None
        self.xmeannc = None
        self.xstdnc = None
        self.x_dim = None
        self.nb_cond_keys = None

    def initialize(self):
        # gpu mode
        self.current_gpu_mode, self.current_gpu_device = get_gpu_device(self.gpu_mode)

        # load the GAN Generator
        self.gaga_params, self.G, _, __ = gaga.load(self.pth_filename, self.gpu_mode)
        self.G = self.G.to(self.current_gpu_device)
        self.G.eval()

        # initialize the z rand
        self.z_rand = gaga_helpers.init_z_rand(self.gaga_params)

        # initialize the mean/std
        self.init_normalization()

    def init_normalization(self):
        params = self.gaga_params
        # normalize the conditional vector
        self.xmean = params["x_mean"][0]
        self.xstd = params["x_std"][0]
        xn = params["x_dim"]
        cn = len(params["cond_keys"])

        # mean and std for cond only
        self.xmeanc = self.xmean[xn - cn: xn]
        self.xstdc = self.xstd[xn - cn: xn]

        # mean and std for non cond
        self.xmeannc = self.xmean[0: xn - cn]
        self.xstdnc = self.xstd[0: xn - cn]

        # which device ?
        dev = self.current_gpu_device
        if self.current_gpu_mode == "mps":
            self.xmean = torch.tensor(self.xmean.astype(np.float32), device=dev)
            self.xstd = torch.tensor(self.xstd.astype(np.float32), device=dev)
        else:
            self.xmean = torch.tensor(self.xmean, device=dev)
            self.xstd = torch.tensor(self.xstd, device=dev)

    def __str__(self):
        mm = gate.g4_units.mm
        s = f"gaga: user gpu mode: {self.gpu_mode}\n"
        s += f"gaga: current gpu mode: {self.current_gpu_mode}"
        s += f"gaga: batch size: {self.batch_size}"
        s += f"gaga: backward: {self.backward_distance/mm} mm"
        return s


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


def generate_spect_images_torch(gaga_source, garf_detector):
    print()
    # init
    gaga_source.initialize()
    garf_detector.initialize()
    # verbose
    print(gaga_source)
    print(garf_detector)
    # create garf planes information

    # create gaga sources
    # loop on batches
    #      generate conditions
    #      generate photon sources
    #      loop on planes
    #           compute intersection
    #           apply garf
    # return images
