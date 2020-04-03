import numpy as np
# import cupy as cp
import torch
from torch.autograd import Variable
import gaga
import datetime
import os
import time
import garf
import gatetools.phsp as phsp
from scipy.stats import kde
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import entropy
from scipy.spatial.transform import Rotation
import SimpleITK as sitk
import logging
logger=logging.getLogger(__name__)


# ----------------------------------------------------------------------------
'''
date format
'''
date_format = "%Y-%m-%d %H:%M:%S"


# ----------------------------------------------------------------------------
def select_keys(x, params, read_keys):
    '''
    Return the selected keys
    '''
    if not 'keys' in params:
        return read_keys, x
    
    keys = params['keys']
    keys = phsp.str_keys_to_array_keys(keys)
    x = phsp.select_keys(x, read_keys, keys)

    return keys, x

# ----------------------------------------------------------------------------
def normalize_data(x):
    '''
    Consider the input vector mean and std and normalize it
    '''
    x_mean = np.mean(x, 0, keepdims=True)
    x_std = np.std(x, 0, keepdims=True)
    x = (x-x_mean)/x_std
    return x, x_mean, x_std

# ----------------------------------------------------------------------------
def init_pytorch_cuda(gpu_mode, verbose=False):
    '''
    Test if pytorch use CUDA. Return type and device
    '''
    
    if (verbose):
        print('pytorch version', torch.__version__)
    dtypef = torch.FloatTensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        if torch.cuda.is_available():
            print('CUDA is available')
        else:
            print('CUDA is *NOT* available')
            
    if (gpu_mode == 'auto'):
        if (torch.cuda.is_available()):
            dtypef = torch.cuda.FloatTensor
    elif (gpu_mode == 'true'):
        if (torch.cuda.is_available()):
            dtypef = torch.cuda.FloatTensor
        else:
            print('Error GPU mode not available')
            exit(0)
    else:
        device = torch.device('cpu');

    if (verbose):
        if (str(device) != 'cpu'):
            print('GPU is enabled')
            print('CUDA version         ', torch.version.cuda)
            print('CUDA device counts   ', torch.cuda.device_count())
            print('CUDA current device  ', torch.cuda.current_device())
            n = torch.cuda.current_device()
            print('CUDA device name     ', torch.cuda.get_device_name(n))
            print('CUDA device address  ', torch.cuda.device(n))
        else:
            print('CPU only (no GPU)')

    return dtypef, device



# ----------------------------------------------------------------------------
def print_network(net):
    '''
    Print info about a network
    '''
    num_params = get_network_nb_parameters(net)
    print(net)
    print('Total number of parameters: %d' % num_params)


# ----------------------------------------------------------------------------
def get_network_nb_parameters(net):
    '''
    Compute total nb of parameters
    '''
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params


# ----------------------------------------------------------------------------
def print_info(params, optim):
    '''
    Print info about a trained GAN-PHSP
    '''
    # print parameters
    for e in params:
        if (e[0] != '#') and (e != 'x_mean') and (e != 'x_std'):
            print('   {:22s} {}'.format(e, str(params[e])))

    # additional info
    try:
        start = datetime.datetime.strptime(params['start date'], gaga.date_format)
        end = datetime.datetime.strptime(params['end date'], gaga.date_format)
    except:
        start = 0
        end = 0
    delta = end-start
    print('   {:22s} {}'.format('Duration', delta))

    d_loss_real = np.asarray(optim['d_loss_real'][-1])
    d_loss_fake = np.asarray(optim['d_loss_fake'][-1])
    d_loss = d_loss_real + d_loss_fake
    g_loss = np.asarray(optim['g_loss'][-1])
    print('   {:22s} {}'.format('Final d_loss', d_loss))
    print('   {:22s} {}'.format('Final g_loss', g_loss))
    if 'd_best_loss' in optim:
        print('   {:22s} {}'.format('d_best_loss', optim['d_best_loss']))
        print('   {:22s} {}'.format('d_best_epoch', optim['d_best_epoch']))

# ----------------------------------------------------------------------------
def load(filename, gpu_mode='auto', verbose=False, epoch=-1):
    '''
    Load a GAN-PHSP
    Output params   = dict with all parameters
    Output G        = Generator network
    Output optim    = dict with information of the training process
    '''
    
    dtypef, device = init_pytorch_cuda(gpu_mode, verbose)
    if (str(device) == 'cpu'):
        nn = torch.load(filename, map_location=lambda storage, loc: storage)
    else:
        nn = torch.load(filename)

    # get elements
    params = nn['params']
    if not 'optim' in nn:
        optim = nn['model'] ## FIXME compatibility --> to remove
    else:
        optim =  nn['optim']

    D_state = nn['d_model_state']
    if epoch == -1:
        G_state = nn['g_model_state']
    else:
        try:
            index = optim['current_epoch'].index(epoch)
        except:
            print(f"Epoch {epoch} is not in the list : {optim['current_epoch']}")
            exit(0)
        G_state = optim['g_model_state'][index]

    # create the Generator
    # cmin, cmax = gaga.get_min_max_constraints(params)
    # cmin = torch.from_numpy(cmin).type(dtypef)
    # cmax = torch.from_numpy(cmax).type(dtypef)
    G = gaga.Generator(params)
    D = gaga.Discriminator(params)
    
    if (str(device) != 'cpu'):
        G.cuda()
        D.cuda()
        params['current_gpu'] = True
    else:
        params['current_gpu'] = False

    G.load_state_dict(G_state)
    D.load_state_dict(D_state)

    return params, G, D, optim, dtypef


# ----------------------------------------------------------------------------
def get_min_max_constraints(params):
    '''
    Compute the min/max values per dimension according to params['keys'] and params['constraints']
    '''
    
    # clamp take normalisation into account
    x_dim = params['x_dim']
    keys = params['keys']
    ckeys = params['constraints']
    cmin = np.ones((1, x_dim)) * -9999 # FIXME min value
    cmax = np.ones((1, x_dim)) *  9999 # FIXME max value
    for k,v in ckeys.items():
        try:
            index = keys.index(k)
            cmin[0,index] = v[0]
            cmax[0,index] = v[1]
        except:
            continue
        
    x_std = params['x_std']
    x_mean = params['x_mean']
    
    cmin = (cmin-x_mean)/x_std
    cmax = (cmax-x_mean)/x_std

    return cmin, cmax
    

# ----------------------------------------------------------------------------
def plot_epoch(ax, params, optim, filename):
    '''
    Plot D loss wrt to epoch
    3 panels : all epoch / first 20% / last 1% 
    '''

    x1 = np.asarray(optim['d_loss_real'])
    x2 = np.asarray(optim['d_loss_fake'])
    #x = -np.add(x1,x2)
    x = -np.asarray(optim['d_loss']) # with grad penalty

    epoch = np.arange(params['start_epoch'], params['end_epoch'], 1)    
    a = ax#[0]
    l = filename
    a.plot(epoch, x, '-', label='D_loss (GP) '+l)
    z = np.zeros_like(x)
    a.set_xlabel('epoch')
    a.plot(epoch, z, '--')
    a.legend()

    return
    a = ax[1]
    n = int(len(x)*0.2) # first 20%
    xc = x[0:n]
    a.plot(epoch, x, '-', label='D_loss '+l)
    z = np.zeros_like(xc)
    a.set_xlabel('epoch')
    a.set_xlim((0,n))
    ymin = np.amin(xc)
    ymax = np.amax(xc)
    a.set_ylim((ymin,ymax))
    a.plot(z, '--')
    a.legend()
    
    a = ax[2]
    n = max(10, int(len(x)*0.01)) # last 1%
    xc = x
    a.plot(xc, '.-', label='D_loss '+l)
    z = np.zeros_like(xc)
    a.set_xlabel('epoch')
    a.set_xlim((len(xc)-n,len(xc)))
    xc = x[len(x)-n:len(x)]
    ymin = np.amin(xc)
    ymax = np.amax(xc)
    a.set_ylim((ymin,ymax))
    a.plot(epoch, z, '--')
    a.legend()

    
# ----------------------------------------------------------------------------
def plot_epoch_wasserstein(ax, optim, filename):
    '''
    Plot wasserstein 
    '''

    y = np.asarray(optim['w_value'])
    x = np.asarray(optim['w_epoch'])
    if len(x) < 1:
        return

    a = ax[0].twinx()
    a.plot(x, y, '-', color='r', label='W')
    a.legend()

    a = ax[1].twinx()
    a.plot(x, y, '-', color='r', label='W')
    a.legend()

    a = ax[2].twinx()
    a.plot(x, y, '.-', color='r', label='W')
    a.legend()

    
# ----------------------------------------------------------------------------
def generate_samples2(params, G, n, batch_size=-1, normalize=False, to_numpy=False):

    z_dim = params['z_dim']

    if params['current_gpu']:
        dtypef = torch.cuda.FloatTensor
    else:
        dtypef = torch.FloatTensor
        
    batch_size = int(batch_size)
    if batch_size == -1:
        batch_size = int(n)
        to_numpy = True
    if batch_size>n:
        batch_size = int(n)

    m = 0
    z_dim = params['z_dim']
    x_dim = params['x_dim']
    rfake = np.empty((0,x_dim))
    while m < n:
        z = Variable(torch.randn(batch_size, z_dim)).type(dtypef)
        fake = G(z)
        # put back to cpu to allow concatenation
        fake = fake.cpu().data.numpy()
        rfake = np.concatenate((rfake, fake), axis=0)
        m = m+batch_size
        if m+batch_size>n:
            batch_size = n-m

    if not normalize:
        x_mean = params['x_mean']
        x_std = params['x_std']
        rfake = (rfake*x_std)+x_mean

    if to_numpy:
        return rfake
    
    return Variable(torch.from_numpy(rfake)).type(dtypef)


# ----------------------------------------------------------------------------
def fig_plot_marginal(x, k, keys, ax, i, nb_bins, color, r=''):
    a = phsp.fig_get_sub_fig(ax,i)
    index = keys.index(k)
    if len(x[0])>1:
        d = x[:,index]
    else:
        d = x
    label = ' {} $\mu$={:.2f} $\sigma$={:.2f}'.format(k, np.mean(d), np.std(d))
    if r != '':
        a.hist(d, nb_bins,
               density=True,
               histtype='stepfilled',
               facecolor=color,
               alpha=0.5,
               range=r,
               label=label)
    else:
        a.hist(d, nb_bins,
               density=True,
               histtype='stepfilled',
               facecolor=color,
               alpha=0.5,
               label=label)
    a.set_ylabel('Counts')
    a.legend()


# ----------------------------------------------------------------------------
def fig_plot_marginal_2d(x, k1, k2, keys, ax, i, nbins, color):
    a = phsp.fig_get_sub_fig(ax,i)
    index1 = keys.index(k1)
    d1 = x[:,index1]
    index2 = keys.index(k2)
    d2 = x[:,index2]
    label = '{} {}'.format(k1, k2)

    ptype = 'scatter'
    # ptype = 'hist'
    
    if ptype == 'scatter':
        a.scatter(d1, d2, color=color,
                  alpha=0.25,
                  edgecolor='none',
                  s=1)        
        
    if ptype == 'hist':
        cmap = plt.cm.Greens
        if color == 'r':
            cmap = plt.cm.Reds
        a.hist2d(d1, d2,
                 bins=(nbins, nbins), 
                 alpha=0.5,
                 cmap=cmap)

    if ptype == 'density':
        x = d1
        y = d2
        print('kde')
        k = kde.gaussian_kde([x,y])
        xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        a.pcolormesh(xi, yi,
                     zi.reshape(xi.shape), 
                     alpha=0.5)     
        
    a.set_xlabel(k1)
    a.set_ylabel(k2)


# ----------------------------------------------------------------------------
def fig_plot_diff_2d(x, y, keys, kk, ax, fig, nb_bins):
    k1 = kk[0]
    k2 = kk[1]
    index1 = keys.index(k1)
    index2 = keys.index(k2)
    x1 = x[:,index1]
    x2 = x[:,index2]
    y1 = y[:,index1]
    y2 = y[:,index2]
    label = '{} {}'.format(k1, k2)

    # compute histo
    H_x, xedges_x, yedges_x = np.histogram2d(x1, x2, bins=nb_bins)
    H_y, xedges_y, yedges_y = np.histogram2d(y1, y2, bins=(xedges_x, yedges_x))

    # make diff
    #H = (H_y - H_x)/H_y
    np.seterr(divide='ignore', invalid='ignore')
    H = np.divide(H_y - H_x, H_y)

    # plot
    X, Y = np.meshgrid(xedges_x, yedges_x)
    im = ax.pcolormesh(X, Y, H.T) #, norm=LogNorm())
    im.set_clim(vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(k1)
    ax.set_ylabel(k2)
    

# ----------------------------------------------------------------------------
def Jensen_Shannon_divergence(x, y, bins, margin=0):

    #margin = 0#.01 # 5%
    r = [np.amin(x), np.amax(x)]
    if r[0]<0:
        r = [r[0]+margin*r[0], r[1]+margin*r[1]]
    else:
        r = [r[0]-margin*r[0], r[1]+margin*r[1]]
    P, bin_edges = np.histogram(x, range=r, bins=bins, density=True)
    Q, bin_edges = np.histogram(y, range=r, bins=bins, density=True)
    
    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))



# ----------------------------------------------------------------------------
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
    while l_current<l:
        
        # directions: matrix [ndim X l]
        directions = np.random.randn(ndim, l_batch_size)
        directions /= np.linalg.norm(directions, axis=0)
        
        # send to gpu if possible
        directions = torch.from_numpy(directions).type(dtypef)
    
        # Projection (Radon) x = [n X ndim], px = [n X L]
        px = torch.matmul(x,directions)
        py = torch.matmul(y,directions)

        # sum wasserstein1D over all directions
        for i in range(l_batch_size):
            lx = px[:,i]
            ly = py[:,i]
            d += wasserstein1D(lx, ly, p)

        l_current += l_batch_size
        if l_current+l_batch_size>l:
            l_batch_size = l-l_current
    
    d = torch.pow(d/l, 1/p)    
    d = d.data.cpu().numpy()
    return d

# ----------------------------------------------------------------------------
def wasserstein1D(x, y, p=1):
    sx, indices = torch.sort(x)
    sy, indices = torch.sort(y)
    z = (sx-sy)
    return torch.sum(torch.pow(torch.abs(z), p))/len(z)


# ----------------------------------------------------------------------------
def init_plane(n, angle, radius):
    '''
    plane_U, plane_V, plane_point, plane_normal
    '''

    n = int(n)
    logger.info(f'Initialisation of plane with radius {radius} ')
    plane_U = np.array([1,0,0])
    plane_V = np.array([0,1,0])
    r = Rotation.from_euler('y', angle, degrees=True)
    plane_U = r.apply(plane_U)
    plane_V = r.apply(plane_V)

    # normal vector is the cross product of two direction vectors on the plane
    plane_normal = np.cross(plane_U, plane_V)
    plane_normal = np.array([plane_normal]*n)

    center = np.array([0,0,-radius])
    center = r.apply(center)
    plane_center = np.array([center,]*n)

    plane = { 'plane_U': plane_U,
              'plane_V': plane_V,
              'rotation': r,
              'plane_normal': plane_normal,
              'plane_center': plane_center }
    return plane    

    

# ----------------------------------------------------------------------------
def project_on_plane(x, plane, image_plane_size_mm, debug=False):
    '''
    Project the x points (Ekine X Y Z dX dY dZ)
    on the image plane defined by plane_U, plane_V, plane_center, plane_normal
    '''

    logger.info(f'Projection of {len(x)} particles on the plane')
    logger.info(f'Plane size is {image_plane_size_mm} mm')

    # shorter variable names

    # n is the normal plane, duplicated n times
    n = plane['plane_normal']

    # c0 is the center of the plane, duplicated n times
    c0 = plane['plane_center']

    # r is the rotation matrix of the plane, according to the current rotation angle (around Y)
    r = plane['rotation']

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
    ndotu = (n*u).sum(-1)             # dot product between normal plane (n) and direction (u)
    si = -(n*w).sum(-1) / ndotu       # dot product between normal plane and vector from plane to point (w)

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
    logger.info(f'Remove negative direction, remains {mnb}/{len(x)}')

    # si is a (nb) size vector, expand it to (nb x 3)
    msi = np.array([msi]*3).T

    # intersection between point-direction and plane
    psi = mp + msi*mu 

    # apply the inverse of the rotation
    ri = r.inv()
    psip = ri.apply(psi) #- offset

    # remove out of plane (needed ??)
    sizex = image_plane_size_mm[0]/2.0
    sizey = image_plane_size_mm[1]/2.0
    mask1 = psip[:,0]<sizex
    mask2 = psip[:,0]>-sizex
    mask3 = psip[:,1]<sizey
    mask4 = psip[:,1]>-sizey
    m = mask1 & mask2 & mask3 & mask4
    psip = psip[m]
    psi = psi[m]
    mp = mp[m]
    mu = mu[m]
    mx = mx[m]
    mc0 = mc0[m]
    nb = len(psip)
    logger.info(f'Remove points that are out of detector, remains {nb}/{len(x)}')

    # reshape results
    pu = psip[:, 0].reshape((nb, 1)) # u
    pv = psip[:, 1].reshape((nb, 1)) # v
    y =  np.concatenate((pu, pv), axis=1) 

    # rotate direction according to the plane
    mup = ri.apply(mu) 
    norm = np.linalg.norm(mup, axis=1, keepdims=True)
    mup = mup / norm
    dx = mup[:,0]
    dy = mup[:,1]

    # FIXME -> clip arcos -1;1 ?

    # convert direction into theta/phi
    # theta is acos(dy)
    # phi is acos(dx)
    theta = np.degrees(np.arccos(dy)).reshape((nb, 1))
    phi = np.degrees(np.arccos(dx)).reshape((nb, 1))
    y = np.concatenate((y, theta), axis=1)
    y = np.concatenate((y, phi), axis=1)

    # concat the E
    E = mx[:,0].reshape((nb, 1))
    data = np.concatenate((y, E), axis=1)

    return data


# ----------------------------------------------------------------------------
def fig_plot_projected(data):
    '''
    Debug plot
    '''

    b = 100
    x = data[:,0]
    y = data[:,1]
    theta = data[:,2]
    phi = data[:,3]
    E = data[:,4]

    f, ax = plt.subplots(2, 2, figsize=(10,10))

    n, bins, patches = ax[0,0].hist(theta, b, density=True, facecolor='g', alpha=0.35)
    n, bins, patches = ax[0,1].hist(phi, b, density=True, facecolor='g', alpha=0.35)
    n, bins, patches = ax[1,0].hist(E*1000, b, density=True, facecolor='b', alpha=0.35)
    ax[1,1].scatter(x, y, color='r', alpha=0.35, s=1)

    ax[0,0].set_xlabel('Theta angle (deg)')
    ax[0,1].set_xlabel('Phi angle (deg)')
    ax[1,0].set_xlabel('Energy (keV)')
    ax[1,1].set_xlabel('X')
    ax[1,1].set_ylabel('Y')

    plt.tight_layout()
    plt.show()

    
# ----------------------------------------------------------------------------
def gaga_garf_generate_image(p):

    # param
    gan_params = p["gan_params"]
    G = p["G"]
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
    while ev<n:

        # Step 1 : GAN
        t1 = time.time()
        logger.info(f'Generating {batch_size} events')
        x = gaga.generate_samples2(gan_params, G, batch_size, gan_batch_size, normalize=False, to_numpy=True)
        logger.info('Computation time: {0:.3f} sec'.format(time.time()-t1))

        # Step 2 : Projection
        t1 = time.time()
        px = gaga.project_on_plane(x, plane, image_plane_size_mm=image_plane_size_mm, debug=debug)
        logger.info('Computation time: {0:.3f} sec'.format(time.time()-t1))

        # Step3 : GARF
        t1 = time.time()
        logger.info(f'Building image with {len(px)} particles')
        img, sq_img = garf.build_arf_image_with_nn(garf_nn, garf_model, px,
                                                   garf_param, verbose=False, debug=debug)
        images.append(img)
        sq_images.append(sq_img)
        logger.info('Computation time: {0:.3f} sec'.format(time.time()-t1))

        ev += batch_size
        pbar.update(batch_size)
        ev = min(ev,n)
        logger.info('')
    
    # sum images
    im_iter = iter(images)
    im = next(im_iter)
    data = sitk.GetArrayFromImage(im)
    for im in im_iter:
        d = sitk.GetArrayViewFromImage(im)
        data += d
    img = sitk.GetImageFromArray(data)
    img.CopyInformation(images[0])

    # sum images
    im_iter = iter(sq_images)
    im = next(im_iter)
    data = sitk.GetArrayFromImage(im)
    for im in im_iter:
        d = sitk.GetArrayViewFromImage(im)
        data += d
    sq_img = sitk.GetImageFromArray(data)
    sq_img.CopyInformation(sq_images[0])

    return img, sq_img

