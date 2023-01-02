import numpy as np
import torch
from torch.autograd import Variable
import gaga_phsp as gaga
import datetime
import time
import garf
import gatetools.phsp as phsp
from scipy.stats import entropy
from scipy.spatial.transform import Rotation
import SimpleITK as sitk
import logging
import sys
import os
from box import Box, BoxList

logger = logging.getLogger(__name__)

"""
date format
"""
date_format = "%Y-%m-%d %H:%M:%S"


def update_params_with_user_options(params, user_param):
    """
    Update the dict 'params' with the options set by the user on the command line
    """
    for up in user_param:
        params[up[0]] = up[1]


def param_check_keys(params, read_keys):
    """
    Consider the 'keys' tag in param
    - if not exist consider read_keys
    - remove '_' if needed
    - convert to keys_list
    - check keys
    """

    # if no keys, just consider the read ones
    if "keys" not in params:
        params["keys"] = read_keys.join(" ")

    # on command line, '#' may be used to replace space, so we put it back
    if "#" in params["keys"]:
        params["keys"] = params["keys"].replace("#", " ")

    # build the list of keys
    params.keys_list = phsp.str_keys_to_array_keys(params["keys"])

    # check
    for k in params.keys_list:
        if k not in read_keys:
            print(f'Error, the key "{k}" does not belong to read keys: {read_keys}')
            exit(0)

    # debug
    # print('keys:      ', params['keys'])
    # print('keys_list: ', params.keys_list)


def check_input_params(params, fatal_on_unknown_keys=True):
    required = [
        "gpu_mode",
        "model",
        "d_layers",
        "g_layers",
        "d_learning_rate",
        "g_learning_rate",
        "optimiser",
        "d_nb_update",
        "g_nb_update",
        "loss",
        "penalty",
        "penalty_weight",
        "batch_size",
        "epoch",
        "d_dim",
        "g_dim",
        "z_dim",
        "r_instance_noise_sigma",
        "f_instance_noise_sigma",
        "activation",
        "z_rand_type",
        "shuffle",
        "keys",
        "epoch_dump",
        "keys_list",
    ]
    automated = [
        "params_filename",
        "training_size",
        "x_dim",
        "progress_bar",
        "training_filename",
        "start_date",
        "hostname",
        "d_nb_weights",
        "g_nb_weights",
        "x_mean",
        "x_std",
        "end_epoch",
        "Duration",
        "duration",
        "end_date",
        "output_filename",
        "cond_keys",
    ]

    # forced
    if "r_instance_noise_sigma" not in params:
        params["r_instance_noise_sigma"] = -1
    if "f_instance_noise_sigma" not in params:
        params["f_instance_noise_sigma"] = -1

    """ 
        WARNING: management of the list of keys
        - user defined value in json is 'keys', but this is a reserve keyword :(
        - also user provide a simple str with space between the keys
        -> we convert into a list of str, named keys_list
        
        When the gan is trained, the params saved in the gan file is already a list. 
        
        Same for cond_keys.
    """

    # automated cond keys
    if "cond_keys" not in params:
        params["cond_keys"] = []
    else:
        if type(params["cond_keys"]) != list and type(params["cond_keys"]) != BoxList:
            params["cond_keys"] = phsp.str_keys_to_array_keys(params["cond_keys"])

    # keys (for backward compatible)
    if "keys_list" not in params:
        if type(params["keys"]) == list:
            params["keys_list"] = params["keys"].copy()
        else:
            params["keys_list"] = phsp.str_keys_to_array_keys(params["keys"])

    # old versions does not have some tags
    if "activation" not in params:
        params["activation"] = "relu"
    if "loss" not in params:
        params["loss"] = "wasserstein"
    if "model" not in params:
        params["model"] = "v3"
    if "penalty" not in params:
        params["penalty"] = "clamp"
    if "penalty_weight" not in params:
        params["penalty_weight"] = 0
    if "z_rand_type" not in params:
        params["z_rand_type"] = "randn"
    if "epoch_dump" not in params:
        params["epoch_dump"] = -1

    # check required
    for req in required:  # + automated
        if req not in params:
            print(f'Error, the parameters "{req}" is required in {params}')
            exit(0)

    # look unknown param
    optional = [
        "start_pth",
        "start_epoch",
        "schedule_learning_rate_step",
        "schedule_learning_rate_gamma",
        "label_smoothing",
        "spectral_norm",
        "epoch_store_model_every",
        "RMSprop_d_momentum",
        "RMSprop_g_momentum",
        "RMSProp_d_alpha",
        "RMSProp_g_alpha",
        "GAN_model",
        "RMSProp_d_weight_decay",
        "RMSProp_g_weight_decay",
        "RMSProp_d_centered",
        "RMSProp_g_centered",
    ]
    for p in params:
        if p[0] == "#":
            continue
        if p in optional:
            # print('Found optional: ', p)
            pass
        else:
            if p not in required + automated:
                print(
                    f'Warning unknown key named "{p}" in the parameters (deprecated?)'
                )
                if fatal_on_unknown_keys:
                    exit(0)

    # special for adam
    if params["optimiser"] == "adam":
        required_adam = ["g_weight_decay", "d_weight_decay", "beta_1", "beta_2"]
        for req in required_adam:
            if req not in params:
                print(f'Error, the Adam parameters "{req}" is required in {params}')
                exit(0)


def normalize_data(x):
    """
    Consider the input vector mean and std and normalize it
    """
    x_mean = np.mean(x, 0, keepdims=True)
    x_std = np.std(x, 0, keepdims=True)
    x = (x - x_mean) / x_std
    return x, x_mean, x_std


def init_pytorch_cuda(gpu_mode, verbose=False):
    """
    Test if pytorch use CUDA. Return type and device
    """

    if verbose:
        print("pytorch version", torch.__version__)
    dtypef = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        if torch.cuda.is_available():
            print("CUDA is available")
        else:
            print("CUDA is *NOT* available")

    if gpu_mode == "auto":
        if torch.cuda.is_available():
            dtypef = torch.cuda.FloatTensor
    elif gpu_mode == "true":
        if torch.cuda.is_available():
            dtypef = torch.cuda.FloatTensor
        else:
            print("Error GPU mode not available")
            exit(0)
    else:
        device = torch.device("cpu")

    if verbose:
        if str(device) != "cpu":
            print("GPU is enabled")
            print("CUDA version         ", torch.version.cuda)
            print("CUDA device counts   ", torch.cuda.device_count())
            print("CUDA current device  ", torch.cuda.current_device())
            n = torch.cuda.current_device()
            print("CUDA device name     ", torch.cuda.get_device_name(n))
            print("CUDA device address  ", torch.cuda.device(n))
        else:
            print("CPU only (no GPU)")

    return dtypef, device


def print_network(net):
    """
    Print info about a network
    """
    num_params = get_network_nb_parameters(net)
    print(net)
    print("Total number of parameters: %d" % num_params)


def get_network_nb_parameters(net):
    """
    Compute total nb of parameters
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params


def print_info(params, optim):
    """
    Print info about a trained GAN-PHSP
    """
    # print parameters
    for e in sorted(params):
        if (e[0] != "#") and (e != "x_mean") and (e != "x_std"):
            print("   {:22s} {}".format(e, str(params[e])))

    # additional info
    try:
        start = datetime.datetime.strptime(params["start date"], gaga.date_format)
        end = datetime.datetime.strptime(params["end date"], gaga.date_format)
    except:
        start = 0
        end = 0
    delta = end - start
    print("   {:22s} {}".format("Duration", delta))

    d_loss_real = np.asarray(optim["d_loss_real"][-1])
    d_loss_fake = np.asarray(optim["d_loss_fake"][-1])
    d_loss = d_loss_real + d_loss_fake
    g_loss = np.asarray(optim["g_loss"][-1])
    print("   {:22s} {}".format("Final d_loss", d_loss))
    print("   {:22s} {}".format("Final g_loss", g_loss))
    if "d_best_loss" in optim:
        print("   {:22s} {}".format("d_best_loss", optim["d_best_loss"]))
        print("   {:22s} {}".format("d_best_epoch", optim["d_best_epoch"]))

    # version
    p = "python"
    v = sys.version.replace("\n", "")
    print(f"   {p:22s} {v}")
    p = "pytorch"
    print(f"   {p:22s} {torch.__version__}")


def print_info_short(params, optim):
    p = Box(params)
    s = (
        f"H {p.params_filename} {p.d_dim} {p.g_dim} L {p.d_layers} {p.g_layers} Z {p.z_dim} "
        f"{p.penalty} {p.penalty_weight} lr {p.d_learning_rate} {p.g_learning_rate} "
    )
    try:
        s += f"sc {p.schedule_learning_rate_step} {p.schedule_learning_rate_gamma} "
    except:
        pass
    s += f"D:G {p.d_nb_update}:{p.g_nb_update} {p.epoch} {p.batch_size} {p.duration}"

    print(s)


def create_G_and_D_model(params):
    G = None
    D = None
    if params["model"] == "v3":
        G = gaga.Generator(params)
        D = gaga.Discriminator(params)
        return G, D
    if not D or not G:
        print("Error in create G and D model, unknown model version?")
        print(params["model"])
        print(params["GAN_model"])
        exit(0)


def auto_output_filename(params, output, output_folder):
    if output != "auto":
        params.output_filename = output
        return
    # create an output name with some params
    b, extension = os.path.splitext(os.path.basename(params.params_filename))
    output = f"{b}_{params.penalty}_{params.penalty_weight}_{params.end_epoch}.pth"
    params.output_filename = os.path.join(output_folder, output)


def load(
        filename, gpu_mode="auto", verbose=False, epoch=-1, fatal_on_unknown_keys=True
):
    """
    Load a GAN-PHSP
    Output params   = dict with all parameters
    Output G        = Generator network
    Output optim    = dict with information of the training process
    """

    dtypef, device = init_pytorch_cuda(gpu_mode, verbose)
    if str(device) == "cpu":
        nn = torch.load(filename, map_location=lambda storage, loc: storage)
    else:
        nn = torch.load(filename)

    # get elements
    params = nn["params"]

    gaga.check_input_params(params, fatal_on_unknown_keys)
    if not "optim" in nn:
        optim = nn["model"]  ## FIXME compatibility --> to remove
    else:
        optim = nn["optim"]

    D_state = nn["d_model_state"]
    if epoch == -1:
        G_state = nn["g_model_state"]
    else:
        try:
            index = optim["current_epoch"].index(epoch)
        except:
            print(f"Epoch {epoch} is not in the list : {optim['current_epoch']}")
            exit(0)
        G_state = optim["g_model_state"][index]

    # create the Generator and the Discriminator (Critic)
    G, D = create_G_and_D_model(params)

    if str(device) != "cpu":
        G.cuda()
        D.cuda()
        params["current_gpu"] = True
    else:
        params["current_gpu"] = False

    G.load_state_dict(G_state)
    D.load_state_dict(D_state)

    return params, G, D, optim, dtypef


def get_min_max_constraints(params):
    """
    Compute the min/max values per dimension according to params['keys'] and params['constraints']
    """

    # clamp take normalisation into account
    x_dim = params["x_dim"]
    keys = params["keys"]
    ckeys = params["constraints"]
    cmin = np.ones((1, x_dim)) * -9999  # FIXME min value
    cmax = np.ones((1, x_dim)) * 9999  # FIXME max value
    for k, v in ckeys.items():
        try:
            index = keys.index(k)
            cmin[0, index] = v[0]
            cmax[0, index] = v[1]
        except:
            continue

    x_std = params["x_std"]
    x_mean = params["x_mean"]

    cmin = (cmin - x_mean) / x_std
    cmax = (cmax - x_mean) / x_std

    return cmin, cmax


def get_RMSProp_optimisers(self, p):
    """
    momentum (float, optional) – momentum factor (default: 0)
    alpha (float, optional) – smoothing constant (default: 0.99)
    centered (bool, optional) – if True, compute the centered RMSProp,
             the gradient is normalized by an estimation of its variance
    weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
    """

    d_learning_rate = p["d_learning_rate"]
    g_learning_rate = p["g_learning_rate"]

    if "RMSprop_d_momentum" not in p:
        p["RMSprop_d_momentum"] = 0
    if "RMSprop_g_momentum" not in p:
        p["RMSprop_g_momentum"] = 0

    if "RMSProp_d_alpha" not in p:
        p["RMSProp_d_alpha"] = 0.99
    if "RMSProp_g_alpha" not in p:
        p["RMSProp_g_alpha"] = 0.99

    if "RMSProp_d_weight_decay" not in p:
        p["RMSProp_d_weight_decay"] = 0
    if "RMSProp_g_weight_decay" not in p:
        p["RMSProp_g_weight_decay"] = 0

    if "RMSProp_d_centered" not in p:
        p["RMSProp_d_centered"] = False
    if "RMSProp_g_centered" not in p:
        p["RMSProp_g_centered"] = False

    RMSprop_d_momentum = p["RMSprop_d_momentum"]
    RMSprop_g_momentum = p["RMSprop_g_momentum"]
    RMSProp_d_alpha = p["RMSProp_d_alpha"]
    RMSProp_g_alpha = p["RMSProp_g_alpha"]
    RMSProp_d_weight_decay = p["RMSProp_d_weight_decay"]
    RMSProp_g_weight_decay = p["RMSProp_g_weight_decay"]
    RMSProp_d_centered = p["RMSProp_d_centered"]
    RMSProp_g_centered = p["RMSProp_g_centered"]

    d_optimizer = torch.optim.RMSprop(
        self.D.parameters(),
        lr=d_learning_rate,
        momentum=RMSprop_d_momentum,
        alpha=RMSProp_d_alpha,
        weight_decay=RMSProp_d_weight_decay,
        centered=RMSProp_d_centered,
    )
    g_optimizer = torch.optim.RMSprop(
        self.G.parameters(),
        lr=g_learning_rate,
        momentum=RMSprop_g_momentum,
        alpha=RMSProp_g_alpha,
        weight_decay=RMSProp_g_weight_decay,
        centered=RMSProp_g_centered,
    )

    return d_optimizer, g_optimizer


def get_z_rand(params):
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


def generate_samples2(
        params,
        G,
        D,
        n,
        batch_size=-1,
        normalize=False,
        to_numpy=False,
        z=None,
        cond=None,
        silence=False,
):
    if params["current_gpu"]:
        dtypef = torch.cuda.FloatTensor
    else:
        dtypef = torch.FloatTensor

    # batch size -> if n is lower, batch size is n
    batch_size = int(batch_size)
    if batch_size == -1:
        batch_size = int(n)
        to_numpy = True

    if batch_size > n:
        batch_size = int(n)

    # get z random (gauss or uniform)
    z_rand = get_z_rand(params)

    # is this a conditional GAN ?
    is_conditional = not cond is None

    # normalize the input condition
    ncond = 0
    if is_conditional:
        # normalize the conditional vector
        xmean = params["x_mean"][0]
        xstd = params["x_std"][0]
        xn = params["x_dim"]
        cn = len(params["cond_keys"])
        ncond = cn
        # mean and std for cond only
        xmeanc = xmean[xn - cn: xn]
        xstdc = xstd[xn - cn: xn]
        # mean and std for non cond
        xmeannc = xmean[0: xn - cn]
        xstdnc = xstd[0: xn - cn]
        # normalize the condition
        cond = (cond - xmeanc) / xstdc
    else:
        if len(params["cond_keys"]) > 0:
            print(
                f'Error : GAN is conditional, you should provide the condition: {params["cond_keys"]}'
            )
            exit(0)

    langevin_latent_sampling_flag = False
    if "langevin_latent_sampling" in params:
        langevin_latent_sampling_flag = True

    m = 0
    z_dim = params["z_dim"]
    x_dim = params["x_dim"]
    rfake = np.empty((0, x_dim - ncond))
    while m < n:
        if not silence:
            print(f"Batch {m}/{n}")
        # no more samples than needed
        current_gpu_batch_size = batch_size
        if current_gpu_batch_size > n - m:
            current_gpu_batch_size = n - m
        # print('(G) current_gpu_batch_size', current_gpu_batch_size)

        # (checking Z allow to reuse z for some special test case)
        # if None == z:
        z = Variable(z_rand(current_gpu_batch_size, z_dim)).type(dtypef)

        # condition ?
        if is_conditional:
            condx = (
                Variable(torch.from_numpy(cond[m: m + current_gpu_batch_size]))
                .type(dtypef)
                .view(current_gpu_batch_size, cn)
            )
            z = torch.cat((z.float(), condx.float()), dim=1)

        # FIXME test langevin
        if langevin_latent_sampling_flag:
            z = gaga.langevin_latent_sampling(G, D, params, z)

        fake = G(z)
        # put back to cpu to allow concatenation
        fake = fake.cpu().data.numpy()
        rfake = np.concatenate((rfake, fake), axis=0)

        m = m + current_gpu_batch_size

    if not normalize:
        x_mean = params["x_mean"]
        x_std = params["x_std"]
        if is_conditional:
            # do not consider the mean/std of the condition part
            x_mean = xmeannc
            x_std = xstdnc
        rfake = (rfake * x_std) + x_mean

    if to_numpy:
        return rfake

    return Variable(torch.from_numpy(rfake)).type(dtypef)


def generate_samples3(params, G, n, cond):
    """
    Like generate_samples2 but with less options, to see if it can be faster

    - batch size is managed elsewhere

    """
    if params["current_gpu"]:
        dtypef = torch.cuda.FloatTensor
    else:
        dtypef = torch.FloatTensor

    # normalize the conditional vector
    xmean = params["x_mean"][0]
    xstd = params["x_std"][0]
    xn = params["x_dim"]
    cn = len(params["cond_keys"])
    ncond = cn
    # mean and std for cond only
    xmeanc = xmean[xn - cn: xn]
    xstdc = xstd[xn - cn: xn]
    # mean and std for non cond
    xmeannc = xmean[0: xn - cn]
    xstdnc = xstd[0: xn - cn]
    # normalize the condition
    cond = (cond - xmeanc) / xstdc

    m = 0
    z_dim = params["z_dim"]
    x_dim = params["x_dim"]
    rfake = np.empty((0, x_dim - ncond))

    # (checking Z allow to reuse z for some special test case)
    # if None == z:
    z = Variable(torch.randn(n, z_dim)).type(dtypef)

    # condition ?
    condx = (
        Variable(torch.from_numpy(cond[m: m + n]))
        .type(dtypef)
        .view(n, cn)
    )
    z = torch.cat((z.float(), condx.float()), dim=1)

    # Go !!!
    fake = G(z)

    # put back to cpu to allow concatenation
    fake = fake.cpu().data.numpy()
    rfake = np.concatenate((rfake, fake), axis=0)

    # do not consider the mean/std of the condition part
    x_mean = xmeannc
    x_std = xstdnc
    rfake = (rfake * x_std) + x_mean

    return rfake


def Jensen_Shannon_divergence(x, y, bins, margin=0):
    # margin = 0#.01 # 5%
    r = [np.amin(x), np.amax(x)]
    if r[0] < 0:
        r = [r[0] + margin * r[0], r[1] + margin * r[1]]
    else:
        r = [r[0] - margin * r[0], r[1] + margin * r[1]]
    P, bin_edges = np.histogram(x, range=r, bins=bins, density=True)
    Q, bin_edges = np.histogram(y, range=r, bins=bins, density=True)

    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def sliced_wasserstein(x, y, l, p=1):
    l = int(l)
    ndim = len(x[0])

    if ndim == 1:
        d = wasserstein1D(x, y, p)
        d = d.data.cpu().numpy()
        return d

    dtypef = torch.FloatTensor
    if x.is_cuda:
        dtypef = torch.cuda.FloatTensor
    l_batch_size = int(1e2)
    l_current = 0
    d = 0
    while l_current < l:

        # directions: matrix [ndim X l]
        directions = np.random.randn(ndim, l_batch_size)
        directions /= np.linalg.norm(directions, axis=0)

        # send to gpu if possible
        directions = torch.from_numpy(directions).type(dtypef)

        # Projection (Radon) x = [n X ndim], px = [n X L]
        px = torch.matmul(x, directions)
        py = torch.matmul(y, directions)

        # sum wasserstein1D over all directions
        for i in range(l_batch_size):
            lx = px[:, i]
            ly = py[:, i]
            d += wasserstein1D(lx, ly, p)

        l_current += l_batch_size
        if l_current + l_batch_size > l:
            l_batch_size = l - l_current

    d = torch.pow(d / l, 1 / p)
    d = d.data.cpu().numpy()
    return d


def wasserstein1D(x, y, p=1):
    sx, indices = torch.sort(x)
    sy, indices = torch.sort(y)
    z = sx - sy
    return torch.sum(torch.pow(torch.abs(z), p)) / len(z)


def init_plane(n, angle, radius):
    """
    plane_U, plane_V, plane_point, plane_normal
    """

    n = int(n)
    logger.info(f"Initialisation of plane with radius {radius} ")
    plane_U = np.array([1, 0, 0])
    plane_V = np.array([0, 1, 0])
    r = Rotation.from_euler("y", angle, degrees=True)
    plane_U = r.apply(plane_U)
    plane_V = r.apply(plane_V)

    # normal vector is the cross product of two direction vectors on the plane
    plane_normal = np.cross(plane_U, plane_V)
    plane_normal = np.array([plane_normal] * n)

    center = np.array([0, 0, -radius])
    center = np.array([0, 0, -radius])
    center = r.apply(center)
    plane_center = np.array(
        [
            center,
        ]
        * n
    )

    plane = {
        "plane_U": plane_U,
        "plane_V": plane_V,
        "rotation": r,
        "plane_normal": plane_normal,
        "plane_center": plane_center,
    }
    # logger.info(f'Initialisation of plane {plane} ')
    return plane


def init_plane2(n, angle, radius, spect_table_shift_mm):
    """
    plane_U, plane_V, plane_point, plane_normal
    """

    n = int(n)
    plane_U = np.array([1, 0, 0])
    plane_V = np.array([0, 1, 0])
    r1 = Rotation.from_euler("x", 90, degrees=True)
    r2 = Rotation.from_euler("z", angle, degrees=True)
    r = r2 * r1
    plane_U = r.apply(plane_U)
    plane_V = r.apply(plane_V)

    # normal vector is the cross product of two direction vectors on the plane
    plane_normal = np.cross(plane_U, plane_V)
    plane_normal = np.array([plane_normal] * n)

    center = np.array([0, -spect_table_shift_mm, -radius])
    center = r.apply(center)
    plane_center = np.array([center] * n)

    plane = {
        "plane_U": plane_U,
        "plane_V": plane_V,
        "rotation": r.inv(),  # [r] * n,
        "plane_normal": plane_normal,
        "plane_center": plane_center,
    }

    return plane


def project_on_plane(x, plane, image_plane_size_mm, debug=False):
    """
    Project the x points (Ekine X Y Z dX dY dZ)
    on the image plane defined by plane_U, plane_V, plane_center, plane_normal
    """

    logger.info(f"Projection of {len(x)} particles on the plane")
    logger.info(f"Plane size is {image_plane_size_mm} mm")

    # shorter variable names

    # n is the normal plane, duplicated n times
    n = plane["plane_normal"][0: len(x)]

    # c0 is the center of the plane, duplicated n times
    c0 = plane["plane_center"][0: len(x)]

    # r is the rotation matrix of the plane, according to the current rotation angle (around Y)
    r = plane["rotation"][0: len(x)]

    # p is the set of points position generated by the GAN
    p = x[:, 1:4]

    # u is the set of points direction generated by the GAN
    u = x[:, 4:7]

    # w is the set of vectors from all points to the plane center
    w = p - c0

    # project to plane
    ## dot product : out = (x*y).sum(-1)
    # https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
    # http://geomalgorithms.com/a05-_intersect-1.html
    # https://github.com/pytorch/pytorch/issues/18027
    ndotu = (n * u).sum(-1)  # dot product between normal plane (n) and direction (u)
    si = (
            -(n * w).sum(-1) / ndotu
    )  # dot product between normal plane and vector from plane to point (w)

    # only positive (direction to the plane)
    mask = si > 0
    mw = w[mask]
    mu = u[mask]
    mc0 = c0[mask]
    mn = n[mask]
    mx = x[mask]
    mp = p[mask]
    msi = si[mask]
    mnb = len(msi)
    logger.info(f"Remove negative direction, remains {mnb}/{len(x)}")

    # si is a (nb) size vector, expand it to (nb x 3)
    msi = np.array([msi] * 3).T

    # intersection between point-direction and plane
    psi = mp + msi * mu

    # apply the inverse of the rotation
    ri = r.inv()
    psip = ri.apply(psi)  # - offset

    # remove out of plane (needed ??)
    sizex = image_plane_size_mm[0] / 2.0
    sizey = image_plane_size_mm[1] / 2.0
    mask1 = psip[:, 0] < sizex
    mask2 = psip[:, 0] > -sizex
    mask3 = psip[:, 1] < sizey
    mask4 = psip[:, 1] > -sizey
    m = mask1 & mask2 & mask3 & mask4
    psip = psip[m]
    psi = psi[m]
    mp = mp[m]
    mu = mu[m]
    mx = mx[m]
    mc0 = mc0[m]
    nb = len(psip)
    logger.info(f"Remove points that are out of detector, remains {nb}/{len(x)}")

    # reshape results
    pu = psip[:, 0].reshape((nb, 1))  # u
    pv = psip[:, 1].reshape((nb, 1))  # v
    y = np.concatenate((pu, pv), axis=1)

    # rotate direction according to the plane
    mup = ri.apply(mu)
    norm = np.linalg.norm(mup, axis=1, keepdims=True)
    mup = mup / norm
    dx = mup[:, 0]
    dy = mup[:, 1]

    # FIXME -> clip arcos -1;1 ?

    # convert direction into theta/phi
    # theta is acos(dy)
    # phi is acos(dx)
    theta = np.degrees(np.arccos(dy)).reshape((nb, 1))
    phi = np.degrees(np.arccos(dx)).reshape((nb, 1))
    y = np.concatenate((y, theta), axis=1)
    y = np.concatenate((y, phi), axis=1)

    # concat the E
    E = mx[:, 0].reshape((nb, 1))
    data = np.concatenate((y, E), axis=1)

    return data


def project_on_plane2(x, plane, image_plane_size_mm):
    """
    Project the x points (Ekine X Y Z dX dY dZ)
    on the image plane defined by plane_U, plane_V, plane_center, plane_normal
    """

    # n is the normal plane, duplicated n times
    n = plane["plane_normal"][0: len(x)]

    # c0 is the center of the plane, duplicated n times
    c0 = plane["plane_center"][0: len(x)]

    # r is the rotation matrix of the plane, according to the current rotation angle (around Y)
    r = plane["rotation"]  # [0: len(x)]

    # p is the set of points position generated by the GAN
    p = x[:, 1:4]  # FIXME indices of the position

    # u is the set of points direction generated by the GAN
    u = x[:, 4:7]  # FIXME indices of the position

    # w is the set of vectors from all points to the plane center
    w = p - c0

    # project to plane
    # dot product : out = (x*y).sum(-1)
    # https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
    # http://geomalgorithms.com/a05-_intersect-1.html
    # https://github.com/pytorch/pytorch/issues/18027

    # dot product between normal plane (n) and direction (u)
    ndotu = (n * u).sum(-1)

    # dot product between normal plane and vector from plane to point (w)
    si = (-(n * w).sum(-1) / ndotu)

    # only positive (direction to the plane)
    mask = si > 0
    mu = u[mask]
    mc0 = c0[mask]
    mx = x[mask]
    mp = p[mask]
    msi = si[mask]
    mnb = len(msi)
    # print(f"Remove negative direction, remains {mnb}/{len(x)}")

    # si is a (nb) size vector, expand it to (nb x 3)
    msi = np.array([msi] * 3).T

    # intersection between point-direction and plane
    psi = mp + msi * mu

    # offset of the head
    psi = psi + c0[:len(psi)]

    # apply the inverse of the rotation
    psip = r.apply(psi)

    # remove out of plane (needed ??)
    sizex = image_plane_size_mm[0] / 2.0
    sizey = image_plane_size_mm[1] / 2.0
    mask1 = psip[:, 0] < sizex
    mask2 = psip[:, 0] > -sizex
    mask3 = psip[:, 1] < sizey
    mask4 = psip[:, 1] > -sizey
    m = mask1 & mask2 & mask3 & mask4
    psip = psip[m]
    mu = mu[m]
    mx = mx[m]
    nb = len(psip)
    # print(f"Remove points that are out of detector, remains {nb}/{len(x)}")

    # reshape results
    pu = psip[:, 0].reshape((nb, 1))  # u
    pv = psip[:, 1].reshape((nb, 1))  # v
    y = np.concatenate((pu, pv), axis=1)

    # rotate direction according to the plane
    mup = r.apply(mu)
    norm = np.linalg.norm(mup, axis=1, keepdims=True)
    mup = mup / norm
    dx = mup[:, 0]
    dy = mup[:, 1]

    # FIXME -> clip arcos -1;1 ?

    # convert direction into theta/phi
    # theta is acos(dy)
    # phi is acos(dx)
    theta = np.degrees(np.arccos(dy)).reshape((nb, 1))
    phi = np.degrees(np.arccos(dx)).reshape((nb, 1))
    y = np.concatenate((y, theta), axis=1)
    y = np.concatenate((y, phi), axis=1)

    # concat the E
    E = mx[:, 0].reshape((nb, 1))
    data = np.concatenate((y, E), axis=1)

    return data


def gaga_garf_generate_image(p):
    # param
    gan_params = p["gan_params"]
    G = p["G"]
    D = p["D"]
    batch_size = p["batch_size"]
    gan_batch_size = p["gan_batch_size"]
    plane = p["plane"]
    image_plane_size_mm = p["image_plane_size_mm"]
    debug = p["debug"]
    garf_nn = p["garf_nn"]
    garf_model = p["garf_model"]
    garf_param = p["garf_param"]
    pbar = p["pbar"]
    n = p["n"]

    ev = 0
    images = []
    sq_images = []
    while ev < n:

        # check generation of the exact nb of samples
        current_batch_size = batch_size
        if current_batch_size > n - ev:
            current_batch_size = n - ev

        # Step 1 : GAN
        t1 = time.time()
        logger.info(f"Generating {current_batch_size} events")
        x = gaga.generate_samples2(
            gan_params,
            G,
            D,
            current_batch_size,
            gan_batch_size,
            normalize=False,
            to_numpy=True,
        )
        # print('batch / x', current_batch_size, len(x))
        logger.info("Computation time: {0:.3f} sec".format(time.time() - t1))

        # Step 2 : Projection
        t1 = time.time()
        px = gaga.project_on_plane(
            x, plane, image_plane_size_mm=image_plane_size_mm, debug=debug
        )
        logger.info("Computation time: {0:.3f} sec".format(time.time() - t1))

        # Step3 : GARF
        # output image expressed in counts/samples (generated samples)
        t1 = time.time()
        logger.info(f"Building image with {len(px)}/{current_batch_size} particles")
        garf_param["N_dataset"] = current_batch_size
        img, sq_img = garf.build_arf_image_with_nn(
            garf_nn, garf_model, px, garf_param, verbose=False, debug=debug
        )
        images.append(img)
        sq_images.append(sq_img)
        logger.info("Computation time: {0:.3f} sec".format(time.time() - t1))

        ev += current_batch_size
        pbar.update(current_batch_size)
        ev = min(ev, n)
        logger.info("")

    # mean images
    im_iter = iter(images)
    im = next(im_iter)
    data = sitk.GetArrayFromImage(im)
    for im in im_iter:
        d = sitk.GetArrayViewFromImage(im)
        data += d
    data = data / len(images)
    img = sitk.GetImageFromArray(data)
    img.CopyInformation(images[0])

    # mean images
    im_iter = iter(sq_images)
    im = next(im_iter)
    data = sitk.GetArrayFromImage(im)
    for im in im_iter:
        d = sitk.GetArrayViewFromImage(im)
        data += d
    data = data / len(sq_images)
    sq_img = sitk.GetImageFromArray(data)
    sq_img.CopyInformation(sq_images[0])

    return img, sq_img


def append_gaussian(data, mean, cov, n, vx=None, vy=None):
    x, y = np.random.multivariate_normal(mean, cov, n).T
    d = np.column_stack((x, y))
    if not vx is None:
        d = np.column_stack((d, vx))
    if not vy is None:
        d = np.column_stack((d, vy))
    if data is None:
        return d
    data = np.vstack((data, d))
    return data
