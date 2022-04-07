import os
import sys
import argparse
import multiprocessing as mp
from functools import partial
from os.path import join
from collections import OrderedDict
sys.path.append('../')

import numpy as np
import pandas as pd
import trackml.dataset
from scipy import optimize

from utils.graph_building_utils import *
from utils.hit_processing_utils import *
from utils.data_utils import *

def parse_args(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('-v', '--verbose', action='store_true')
    add_arg('-i', '--input-dir', type=str, default='/Volumes/Untitled/trackML/train_1')#'/tigress/jdezoort/codalab/train_1')
    add_arg('-o', '--output-dir', type=str, default='particle_properties')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--redo', type=bool, default=False)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('--task', type=int, default=0)
    return parser.parse_args(args)

def calc_radii(xc, yc, x=[], y=[]):
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def radii_diffs(c, x=[], y=[]):
    radii = calc_radii(*c, x=x, y=y)
    return radii - np.mean(radii)

def calc_circle_pt(R):
    return 0.0003*2*R

def calc_circle_d0(xc, yc, R):
    return (R-np.sqrt(xc**2 + yc**2))

def radius_error(x, y, xc, yc, R):
    angles = np.arctan2(y-yc, x-xc)
    cx = xc + R*np.cos(angles)
    cy = yc + R*np.sin(angles)
    sum2_errs = (x-cx)**2 + (y-cy)**2
    return np.sqrt(np.sum(sum2_errs))

def rotate(u, v, theta):
    rot_mat = np.array([[np.cos(theta), np.sin(theta)], 
                        [-np.sin(theta), np.cos(theta)]])
    uv = np.stack((u, v), axis=1).T
    uv = rot_mat @ uv
    ur, vr = uv[0], uv[1]
    return ur, vr

def regress(x, y, quadratic=False):
    X = np.stack((np.ones(len(x)), x), axis=1)
    if quadratic:
        X = np.stack((np.ones(len(x)), x, x**2), axis=1)
    sol = np.linalg.inv(X.T @ X) @ X.T @ y
    sol = sol.reshape(len(sol))
    return(sol)

def calc_conformal_pt(fit):
    _, _, a, b, R, _, _ = fit
    return 0.0003*2*R
    
def calc_conformal_d0(fit):
    # TODO! fit[2] is not what it was for gage
    _, _, a, b, R, _, _ = fit
    return -fit[2]*(b/np.sqrt(a**2+b**2))**3

def track_fit_plot(x, y, u, v, conformal_fit, xc, yc, R,
                   label=''):
    fig, axs = plt.subplots(ncols=2, dpi=100, figsize=(16,8))
    axs[0].scatter(x, y, label=label, marker='.', s=40)
    phi = np.linspace(0, 2*np.pi, 1000)
    axs[0].plot(R*np.cos(phi)+xc, R*np.sin(phi)+yc, 
            lw=0.5, ls='-', marker='')
    axs[1].scatter(u, v, label=label, marker='.', s=40)
    s = np.linspace(-0.06, 0.06, 1000)
    axs[1].plot(s, conformal_fit[2]*s**2 + conformal_fit[1]*s + conformal_fit[0], 
                lw=0.5, ls='-', marker='')
    plt.legend(loc='best')
    axs[1].set_xlabel('$x$ [m]')
    axs[0].set_ylabel('$y$ [m]')
    axs[0].set_xlim([-150, 150])
    axs[0].set_ylim([-150, 150])
    axs[0].plot(0, 0, ms=20, marker='+',color='black')
    axs[1].set_xlim([-0.06, 0.06])
    axs[1].set_ylim([-0.06, 0.06])
    axs[1].set_xlabel('$u$ [1/mm]')
    axs[1].set_ylabel('$v$ [1/mm]')
    plt.tight_layout()
    plt.show()

def parabolic(u, a, b, R):
    '''
    Due to current error estimation methods, the original formula for the parabolic model is in use
    TODO update this & the error forumula for the newer model
    '''
    epsilon = R - np.sqrt(a**2 + b**2)
    v = 1/(2*b) - u*a*(1/b) - u**2*epsilon*(R/b)**3
    return v

def parabolic_2(u, a, b_inv, R):
    '''
    Due to current error estimation methods, the original formula for the parabolic model is in use
    TODO update this & the error forumula for the newer model
    '''
    epsilon = R - np.sqrt(a**2 + (1/b_inv)**2)
    v = b_inv/2 - u*a*b_inv - u**2*epsilon*(R*b_inv)**3
    return v

def rotate_conformal(u, v, alpha):
    r = np.sqrt(u**2 + v**2)
    theta = alpha + np.arctan2(v, u)
    return r*np.cos(theta), r*np.sin(theta)

def three_stage_fitting(u, v, rt, verbose=False) -> [float, float, float, np.array]:
    """
    Performs direct linear fit first; if it succeeds, uses the resulting slope to obtain rotation angle.
    The track is then rotated to be roughly parallel to the x-axis.
    Another linear fit is performed and the resulting slope and intercept are finally used to initialize
    the curve_fit procedure of the parabolic fit.
    Currently accounts for the cutoff variable i.e. only fits to hits detected in the layers given by n_layers_fit
    Input:
    * u --> u[:cutoff]
    * v --> v[:cutoff]
    Returns a, b, R, pcov
    """

    maxfev = 10000 # 5000 # Maximum iterations of scipy curvefit --> potentially worth optimising

    def linear_direct(u, slope, intercept):
        return slope * u + intercept

    try:  # linear fit for rotation
        lin_fit_params, _ = optimize.curve_fit(linear_direct, u, v)
        slope, intercept = lin_fit_params
        if verbose: print('Linear naive fit params: Slope: ', slope, 'Intercept:', intercept)
    except RuntimeError as exc:
        if verbose: print("Linear fitting failed: " + str(exc))
        if verbose: print("Will try parabolic fit anyway.")
        slope, intercept = 0, 0.01

    b = 1/(2*intercept)
    a = -b*slope

    # Rotation
    alpha = -np.arctan(slope)
    u, v = rotate_conformal(u, v, alpha)

    try:  # linear fit
        lin_fit_params, _ = optimize.curve_fit(linear_direct, u, v)
        slope, intercept = lin_fit_params
        if verbose: print('Linear naive fit params: Slope: ', slope, 'Intercept:', intercept)
    except RuntimeError as exc:
        if verbose: print("Linear fitting failed: " + str(exc))
        if verbose: print("Will try parabolic fit anyway.")
        slope, intercept = 0, 0.01

    b = 1 / (2 * intercept)
    a = -b * slope

    # Parabolic fitting

    try:  # parabolic fit
        # Uses b_inv = 1/b
        b_inv = 1 / b
        fit_params, pcov = optimize.curve_fit(parabolic_2, u, v,
                                                    p0=(np.sqrt(rt-b**2), b_inv, np.sqrt(a ** 2 + b ** 2)), maxfev=maxfev) #,
        a, b_inv, R = fit_params
        b = 1/b_inv #TODO this messes with pcov doesn't it?
        return u, v, a, b, R, pcov, alpha
    except RuntimeError as exc:
        print("Parabolic fitting failed: " + str(exc))
        return u, v, None, None, None, None, None

def parabolic_full(u, a, b, R):
    '''
    Full parabolic fitting model
    '''
    delta = R**2 - a**2 - b**2
    v = 1/(2*b)*(1-delta/(4*b**2)) - u*a/b*(1-delta/(2*b**2)) - u**2*delta*(R**2)/(2*b**3)
    return v

def four_stage_fitting(u, v, three_stage_fit, maxfev=5000):
    '''
    If three stage fitting succeeded, then perhaps the accuracy of the results may be improved by using the full
    parabolic model
    Input:
    * u and v - conformal coordinates of the track hits
    * three_stage_fit - output [a, b, R, pcov, alpha] of the three_stage_fitting process
    * maxfev (optional) - maximum iterations of scipy optimise
    '''
    print("attempting full parabolic fit...")
    a, b, R, pcov, alpha = three_stage_fit
    try:
        fit_params, pcov = optimize.curve_fit(parabolic_full, u, v,
                                                    p0=(a, b, R), maxfev=maxfev) #,
        a, b, R = fit_params
        return u, v, a, b, R, pcov, alpha
    except RuntimeError as exc:
        # If the fit failed, return the parameters found by the three-stage fit
        print("Full parabolic fitting failed; using the simplified parabolic fit parameters: " + str(exc))
        return u, v, a, b, R, pcov, alpha

def make_df(prefix, output_dir, endcaps=True,
            remove_noise=False, remove_duplicates=False,
            n_layers_fit=4):
    
    # define valid layer-layer connections
    layer_pairs = [(0,1), (1,2), (2,3)]                 # barrel-barrel
    layer_pairs.extend([(0,4), (1,4), (2,4), (3,4),     # barrel-LEC
                        (0,11), (1,11), (2,11), (3,11), # barrel-REC
                        (4,5), (5,6), (6,7),            # LEC-LEC
                        (7,8), (8,9), (9,10),
                        (11,12), (12,13), (13,14),      # REC-REC
                        (14,15), (15,16), (16,17)])
    valid_connections = set(layer_pairs)

    # load the data
    evtid = int(prefix[-9:])
    print(evtid)
    logging.info('Event %i, loading data' % evtid)
    hits, particles, truth = trackml.dataset.load_event(
        prefix, parts=['hits', 'particles', 'truth'])
    hits = hits.assign(evtid=evtid)

    # apply hit selection
    logging.info('Event %i, selecting hits' % evtid)
    hits, particles = select_hits(hits, truth, particles, 0, endcaps,
                                  remove_noise, remove_duplicates)

    # get truth information for each particle
    hits_by_particle = hits.groupby('particle_id')
    df_properties = []

    # Reporting success rate
    successful_fits = 0
    unsuccessful_fits = 0
    for i, (particle_id, particle_hits) in enumerate(hits_by_particle):
        properties = pd.DataFrame({'particle_id': particle_id, 'pt_true': 0, 'eta_pt': 0,
                                   'd0': 0, 'pt_fit': 0, 'q': 0, 'n_track_segs': 0,
                                   'n_layers_hit': 0, 'n_hits': len(particle_hits),
                                   'reconstructable': True, 'skips_layer': False,
                                   'good_fit': False, 'anomalous': False,
                                   'pt_err': 0}, index=[i])
        
        # explicit noise case
        if (particle_id==0): 
            properties['reconstructable'] = False
            df_properties.append(properties)
            continue
        
        # fill in properties of real particles
        properties['pt_true'] = particle_hits['pt'].values[0]
        properties['eta_pt'] = particle_hits['eta_pt'].values[0]
        properties['q'] = particle_hits['q'].values[0]
        layers_hit = particle_hits.layer.values
        hits_per_layer = Counter(layers_hit) # dict of layer: nhits
        hits_per_layer = OrderedDict(hits_per_layer)
        unique_layers_hit = np.unique(layers_hit)
        properties['q'] = particle_hits['q'].values[0]
        properties['n_layers_hit'] = len(unique_layers_hit)
        
        # implicit noise (single-layer particles)
        if (len(unique_layers_hit)==1):
            properties['reconstructable'] = False
            df_properties.append(properties)
            continue
            
        # now particles have hit >1 layer 
        paired_layers = set(zip(unique_layers_hit[:-1], 
                                unique_layers_hit[1:]))
        skips_layer = not paired_layers.issubset(valid_connections)
        
        # figure out how many possible track segments to capture
        good_layer_pairs = paired_layers.intersection(valid_connections)
        edge_counts = [hits_per_layer[lp[0]] * hits_per_layer[lp[1]]
                       for lp in good_layer_pairs]
        edge_count = np.sum(edge_counts)
        properties['n_track_segs'] = edge_count
        
        # particles that skipped a layer
        if (skips_layer):
            properties['skips_layer'] = True
            properties['reconstructable'] = False
            df_properties.append(properties)
            continue
            
        # two-layer particles
        if (len(unique_layers_hit)==2):
            properties['reconstructable'] = False
            df_properties.append(properties)
            continue
        
        # coordinates for track fits
        true_pt = properties['pt_true'].values[0]
        layer_id = particle_hits['layer'].values
        sort_idx = np.argsort(layer_id)
        x = particle_hits['x'].values[sort_idx]
        y = particle_hits['y'].values[sort_idx]
        vx = particle_hits['vx'].values[0] # force vertex into fit TODO
        vy = particle_hits['vy'].values[0]
        x = np.insert(x, 0, vx)
        y = np.insert(y, 0, vy)
        # Transformation to conformal space
        xy2 = x**2 + y**2
        u, v = x/xy2, y/xy2
        
        # fit only the first n_layer_fit layers
        cutoff = np.sum([nhits for l, (lid, nhits) 
                         in enumerate(hits_per_layer.items())
                         if l < n_layers_fit]) + 1
        
        # rotate the conformal coordinates 
        theta = np.arctan2(v[:cutoff][-1]-v[0], u[:cutoff][-1]-u[0])
        ur, vr = rotate(u[:cutoff], v[:cutoff], theta)

        # perform conformal fit - for three_stage_fitting the rotation is performed inside the function
        # and must stay that way as the rotation angle informs the inital linear fit.
        fit = three_stage_fitting(u[:cutoff], v[:cutoff], rt=true_pt/0.0006, verbose=False)

        if (fit[4]==None or fit[3]==None):
            unsuccessful_fits += 1
        else:
            # Attempting a four-stage fit:
            fit_2 = four_stage_fitting(u[:cutoff], v[:cutoff], three_stage_fit=fit[2:])
            conformal_pt = calc_conformal_pt(fit_2)
            conformal_pt_err = abs((true_pt - conformal_pt) / (true_pt))  # TODO uncertainties
            conformal_d0 = calc_conformal_d0(fit_2)
            properties['d0'] = conformal_d0
            properties['pt_err'] = conformal_pt_err
            properties['pt_fit'] = conformal_pt
            #print(properties['pt_fit'])
            df_properties.append(properties)
            successful_fits += 1


        '''
                # use conformal fit to inform circle fit
        yc_est = 1/(2 * fit[0])
        xc_est = -fit[1]*yc_est
        est = rotate([xc_est], [yc_est], -theta)
        xc_est, yc_est = est[0][0], est[1][0]
        R_est = true_pt/(2*0.0003)
        (xc, yc), ier = optimize.leastsq(radii_diffs, (xc_est, yc_est), args=(x, y))
        R = np.mean(calc_radii(xc, yc, x=x[:cutoff], y=y[:cutoff]))
        circle_pt = calc_circle_pt(R)
        circle_pt_err = abs(true_pt-circle_pt)/true_pt
        circle_d0 = calc_circle_d0(xc, yc, R)
        circle_R_err = radius_error(x, y, xc, yc, R)

        # try the fit again with fewer hits if pt error is bad
        min_err = min(conformal_pt_err, circle_pt_err)
        properties['pt_err'] = min_err
        
        # there was no hope for these fits
        if circle_R_err > 5:
            properties['reconstructable'] = False
            properties['anomalous'] = True
            df_properties.append(properties)
            continue
            
        # otherwise we have a good track and maybe a good fit
        if min_err < 0.5:
            properties['good_fit'] = True    
        if circle_pt_err < conformal_pt_err:
            properties['d0'] = circle_d0
        else:
            properties['d0'] = conformal_d0
        df_properties.append(properties)
        '''
        
    df = pd.concat(df_properties)
    outfile = join(output_dir, f'{evtid}.csv')
    logging.info(f'Writing {outfile}')
    df.to_csv(outfile, index=False)

    # Reporting success rate
    print(f"There were {successful_fits} successful and {unsuccessful_fits} unsuccessful fits performed.")
    return 1

def main(args):
    initialize_logger()
    args = parse_args(args)
    logging.info(f'Args {args}')
    initialize_logger(verbose=args.verbose)
    input_dir = args.input_dir
    output_dir = args.output_dir
    # If the output directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_idx = int(input_dir.split('train_')[-1][0])
    logging.info(f'Running on data from {input_dir}.')
    file_prefixes = get_file_prefixes(input_dir,
                                      n_tasks=args.n_tasks, 
                                      task=args.task, evtid_min=0,
                                      evtid_max=1000,
                                      codalab=False)
    
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(make_df,
                               output_dir=output_dir)
        output = pool.map(process_func, file_prefixes)
        
    logging.info('All done!')


if __name__=='__main__':
    main(sys.argv[1:])
