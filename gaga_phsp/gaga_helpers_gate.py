import numpy as np
import gaga_phsp as gaga
import garf
from garf.helpers import get_gpu_device
from scipy.spatial.transform import Rotation
import itk
import opengate.sources.gansources as gansources
from tqdm import tqdm


def voxelized_source_generator(source_filename):
    gen = gansources.VoxelizedSourceConditionGenerator(
        source_filename, use_activity_origin=True
    )
    gen.compute_directions = True
    return gen.generate_condition


def gaga_garf_generate_spect_initialize(gaga_user_info, garf_user_info):
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


def gaga_garf_generate_spect(
    gaga_user_info, garf_user_info, n, angle_rotations, verbose=True
):
    # n must be int
    n = int(n)

    # allocate the initial list of images :
    # number of angles x number of energy windows x image size
    nbe = garf_user_info.nb_energy_windows
    size = garf_user_info.image_size
    spacing = garf_user_info.image_spacing
    data_size = [len(angle_rotations), nbe, size[0], size[1]]
    data_img = np.zeros(data_size, dtype=np.float64)

    # verbose
    if verbose:
        print(f"GAGA pth                 = {gaga_user_info.pth_filename}")
        print(f"GARF pth                 = {garf_user_info.pth_filename}")
        print(f"GARF hist slice          = {garf_user_info.hit_slice}")
        print(f"Activity source          = {gaga_user_info.activity_source}")
        print(f"Number of energy windows = {garf_user_info.nb_energy_windows}")
        print(f"Image plane size (pixel) = {garf_user_info.image_size}")
        print(f"Image plane spacing (mm) = {spacing}")
        print(f"Image plane size (mm)    = {garf_user_info.image_plane_size_mm}")
        print(f"Number of angles         = {len(angle_rotations)}")
        print(f"GAGA batch size          = {gaga_user_info.batch_size:.1e}")
        print(f"GARF batch size          = {garf_user_info.batch_size:.1e}")
        print(
            f"GAGA GPU mode            = {gaga_user_info.gaga_params['current_gpu_mode']}"
        )
        print(
            f"GARF GPU mode            = {garf_user_info.nn.model_data['current_gpu_mode']}"
        )

    # create the planes for each angle (with max number of values = batch_size)
    planes = []
    gaga_user_info.batch_size = int(gaga_user_info.batch_size)
    projected_points = [None] * len(angle_rotations)
    for rot in angle_rotations:
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
    nb_hits_on_plane = [0] * len(angle_rotations)
    nb_detected_hits = [0] * len(angle_rotations)
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
        for i in range(len(angle_rotations)):
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
    for i in range(len(angle_rotations)):
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
        for i in range(len(angle_rotations)):
            print(f"Angle {i}, nb of hits on plane = {nb_hits_on_plane[i]}")
            print(f"Angle {i}, nb of detected hits = {nb_detected_hits[i]}")

    # Remove first slice (nb of hits)
    if not garf_user_info.hit_slice:
        data_img = data_img[:, 1:, :]

    # Final list of images
    images = []
    for i in range(len(angle_rotations)):
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
    position = x[:, pos_index : pos_index + 3]
    direction = x[:, dir_index : dir_index + 3]
    x[:, pos_index : pos_index + 3] = (
        position - gaga_user_info.backward_distance * direction
    )

    return x
