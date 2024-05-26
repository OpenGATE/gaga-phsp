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


class VoxelizedSourcePDFSamplerTorch:
    """
    This is an alternative to GateSPSVoxelsPosDistribution (c++)
    It is needed because the cond voxel source is used on python side.
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


class GagaSource:
    """
    Used to generate SPECT images out of gate (standalone)
    """

    def __init__(self):
        mm = gate.g4_units.mm
        # user input
        self.gpu_mode = "auto"
        self.pth_filename = None
        self.activity_filename = None
        self.batch_size = 1e5
        self.backward_distance = 400 * mm
        self.energy_threshold_MeV = 0
        self.hit_slice_flag = False

        # translation used for condition sampling (vox source)
        self.cond_translation = None

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
        s += f"gaga translation conditional: {self.cond_translation}\n"
        s += f"gaga nb conditions: {self.nb_cond_keys}\n"
        s += f"gaga x dim: {self.x_dim}\n"
        return s

    def initialize(self):
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
        sp = garf_detector.image_spacing
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
            sp = np.array([sp[0], sp[1], 1.0])
            origin = [
                -size[0] * sp[0] / 2 + sp[0] / 2,
                -size[1] * sp[1] / 2 + sp[1] / 2,
                0,
            ]
            img.SetOrigin(origin)
            img.SetSpacing(sp)
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
                fake = self.generate_particles_torch(current_batch_size)
                current_n += current_batch_size

                # project on detector plane
                garf_detector.project_to_planes_torch(fake)

                # progress bar
                pbar.update(current_batch_size)

        images = garf_detector.save_projections()
        return images

    def generate_particles_torch(self, n):
        # get the conditions
        vox_cond = self.cond_activity.generate_conditions_torch(n)
        vox_cond = vox_cond.to(self.current_gpu_device)
        # go !
        fake = self.G(vox_cond)
        # un-normalize
        fake = (fake * self.x_std_non_cond) + self.x_mean_non_cond

        # Remove particle with too low energy
        # n = len(fake)
        fake = fake[fake[:, 0] > self.energy_threshold_MeV]
        # print(f'Remove E<{self.energy_threshold_MeV} MeV : {len(fake)}/{n}')

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
        fake = gaga.generate_samples3(
            self.gaga_params,
            self.G,
            n=n,
            cond=cond,
        )

        # Remove particle with too low energy
        # n = len(fake)
        fake = fake[fake[:, 0] > self.energy_threshold_MeV]
        # print(f'Remove E<{self.energy_threshold_MeV} MeV : {len(fake)}/{n}')

        # move backward
        fake[:, 1:4] = fake[:, 1:4] - self.backward_distance * fake[:, 4:7]

        return fake


class GagaConditionalVoxelizedActivity:

    def __init__(self, gaga_source):
        self.gaga_source = gaga_source

        # members
        self.sampler = None
        self.translation = None
        self.source_size = None
        self.source_spacing = None

        # init
        self.initialize()

    def initialize(self):
        # read activity image
        source = itk.imread(self.gaga_source.activity_filename)
        source_array = itk.array_from_image(source)

        # device
        dev = self.gaga_source.current_gpu_device

        # compute the offset
        self.source_size = np.array(source_array.shape)
        self.source_spacing = np.array(source.GetSpacing())
        self.translation = (self.source_size - 1) * self.source_spacing / 2
        if self.gaga_source.current_gpu_mode == 'mps':
            self.translation = self.translation.astype(np.float32)
        self.translation = torch.from_numpy(self.translation).to(dev).flip(0)

        # voxelized source sampling
        self.sampler = VoxelizedSourcePDFSamplerTorch(source, dev)

    def generate_conditions_torch(self, n):
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

        # FIXME t ??? # FIXME # FIXME  # FIXME  # FIXME  # FIXME  # FIXME  # FIXME  # FIXME
        t = torch.tensor([5.99999588, -28.00000624, -402.0000022], device=dev)
        p = torch.column_stack((z, y, x)) - self.translation + t
        print(f"DEBUG HERE {self.translation=}")

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
