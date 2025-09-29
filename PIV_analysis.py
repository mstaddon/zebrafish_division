#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:47:10 2023

@author: staddon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit


pix_to_um = 0.39

def plot_piv(exp, duration, interval, vmax=10, pix_to_um=0.39):
    # Track maximum soeed so we give all images the same colour scale
    speed_maxs = []

    for t in range(1, duration + 1):
    
        file = exp+'PIV_output/PIVlab_{:04d}.txt'.format(t)
        
        data = pd.read_csv(file, delimiter=',')
        
        # Rename columns for convinience
        data.columns = ['x', 'y', 'u', 'v', 'type']
        
            
        data['x'] *= pix_to_um
        data['y'] *= pix_to_um
        data['u'] *= pix_to_um / (interval / 60)
        data['v'] *= pix_to_um / (interval / 60)

        
        # Turn data into matrix to colour the background
        matrix = data.groupby(['x', 'y']).mean().unstack()
        u, v = matrix['u'], matrix['v']
        
        speed = (u ** 2 + v ** 2) ** 0.5
        
        speed_maxs.append(speed.values.max())
        
        plt.imshow(np.flipud(speed.T), cmap='viridis',
                    extent=[data['x'].min(), data['x'].max(),
                            data['y'].min(), data['y'].max()],
                    interpolation='bicubic',
                    vmin=0, vmax=vmax)
        
        plt.title('t = {:.2f} mins'.format(t * interval / 60))
        
        # Show arrows
        plt.quiver(data['x'], data['y'], data['u'], data['v'],
                    color='w', width=0.005,
                    scale=10 * vmax)
        plt.axis('equal')
        plt.gcf().set_size_inches(5, 8)
        plt.gcf().savefig(exp+'PIV_output/piv_image_{:06d}.png'.format(t-1),
                          bbox_inches='tight')
        plt.show()
        
        



def fit_exponential_decay(y, v):
    
    # Group and fit exponential
    def exponential(x, l, a):
        return a * np.exp(-np.abs(x) / l)
    
    popt, pcov = curve_fit(exponential, y, v, p0=[100, 1])
    
    # print(popt)
    # x = np.linspace(-30, 30)
    # plt.scatter(y, v)
    # plt.plot(x, exponential(x, popt[0]))
    # plt.ylim(0, 1.1)
    # plt.show()
    
    return popt[0], popt[1]


def process_file(file, angle, interval, pix_interval):
    data = pd.read_csv(file, delimiter=',')
    
    # Rename columns for convinience
    data.columns = ['x', 'y', 'u', 'v', 'type']
    
    data['x'] *= pix_to_um
    data['y'] *= pix_to_um
    data['u'] *= pix_to_um / (interval / 60)
    data['v'] *= pix_to_um / (interval / 60)
    
    data=data[data['type'] == 1]
    
    # Apply rotation matrix
    if angle == 90 or angle == -90:
        
        data['x'], data['y'] = -1.0 * data['y'], 1.0 * data['x']
        data['u'], data['v'] = -1.0 * data['v'], 1.0 * data['u']
        
    elif angle != 0:
        rotation = np.deg2rad(angle)
        sin, cos = np.sin(rotation), np.cos(rotation)
        data['x'], data['y'] = cos * data['x'] - sin * data['y'], sin * data['x'] + cos * data['y']
        data['u'], data['v'] = cos * data['u'] - sin * data['v'], sin * data['u'] + cos * data['v']
    
        # Round to nearest pixel
        data['y'] = np.around(data['y'] / pix_to_um / pix_interval) * pix_to_um * pix_interval
        
    return data


def check_angle(exp, t, angle, center, width=25, pix_interval=30):
    file = exp+'/PIV_output/PIVlab_{:04d}.txt'.format(t)
    
    interval = 1
    
    data = process_file(file, angle, interval, pix_interval)
    
    # Subset data to contain the band
    sub = data[(data['x'] >= center - width) & (data['x'] <= center + width)]
    
    plt.quiver(data['x'], data['y'], data['u'], data['v'])
    plt.axvline(center)
    plt.axis('equal')
    plt.gcf().set_size_inches(8, 8)
    plt.show()
    
    # Subset data to contain the band
    sub = data[(data['x'] >= center - width) & (data['x'] <= center + width)]

    # Average y velocity over y positions
    v_mean = sub.groupby('y')['v'].mean()
    
    
    plt.plot(v_mean)
    plt.xlabel('y')
    plt.ylabel('Mean Velocity')
    plt.show()
    
    if angle == 0:
        print(data['x'].unique())
        
        
def get_hydro_decay(exp, t, interval,
                    center, width,
                    angle=0, pix_interval=32):
    file = exp+'PIV_output/PIVlab_{:04d}.txt'.format(t)

    interval = 1
    data = process_file(file, angle, interval, pix_interval)

    v = (data['u'] ** 2 + data['v'] ** 2) ** 0.5
    
    # Subset data to contain the band
    sub = data[(data['x'] >= center - width) & (data['x'] <= center + width)]

    # Average y velocity over y positions
    v_mean = sub.groupby('y')['v'].mean()

    # Highest velocity point
    x0 = center
    y0 = v_mean.index[v_mean.argmax()]
    
    r = ((data['x'] - x0) ** 2 + (data['y'] - y0) ** 2) ** 0.5
    
    plt.scatter(r, v)
    plt.show()


def flow_and_growth_analysis(exp, duration, interval,
                             center, width,
                             analysis_ranges,
                             angle=0, pix_interval=32):
    
    
    fig = plt.figure(constrained_layout=True)

    gs = GridSpec(2, 6, figure=fig)
    
    # Make axis layout with 2 on top and 3 on bottom
    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 3:])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax5 = fig.add_subplot(gs[1, 4:])
    
    # Get location and speeds of maximum positive and negative velocity
    max_v= []
    max_y = []
    min_v = []
    min_y = []
    
    
    # Spatial profiles
    v_vs_y_max = []
    v_vs_y_min = []
    
    
    ts = []
    for t in range(1, duration + 1, 1):

        file = exp+'PIV_output/PIVlab_{:04d}.txt'.format(t)

        data = process_file(file, angle, interval, pix_interval)
 
        # Subset data to contain the band
        sub = data[(data['x'] >= center - width) & (data['x'] <= center + width)]

        # Average y velocity over y positions
        v_mean = sub.groupby('y')['v'].mean()

        if len(v_mean) == 0:
            continue
    
        max_v.append(v_mean.max())
        max_y.append(v_mean.index[v_mean.argmax()])
        min_v.append(v_mean.min())
        min_y.append(v_mean.index[v_mean.argmin()])
        ts.append(t * interval / 60)
        
        # Fit hydro dynamic length scales
        vs_max = v_mean[np.abs(v_mean.index - max_y[-1]) < 50]
        vs_max.index -= max_y[-1]
        vs_max /= vs_max.max()
        vs_max = vs_max.reset_index()
        vs_max['t'] = t
        
        vs_min = v_mean[np.abs(v_mean.index - min_y[-1]) < 50]
        vs_min.index -= min_y[-1]
        vs_min /= vs_min.min()
        vs_min = vs_min.reset_index()
        vs_min['t'] = t
        
        v_vs_y_max.append(vs_max)
        v_vs_y_min.append(vs_min)
        
    v_vs_y_max = pd.concat(v_vs_y_max)
    v_vs_y_min = pd.concat(v_vs_y_min)
        

    df = pd.DataFrame({'t': ts,
                      'max_v': max_v,
                      'max_y': max_y,
                      'min_v': min_v,
                      'min_y': min_y})
    

    # Show speed and locations over time 
    # plt.fill_between([26.5 * interval / 60, 30.5 * interval / 60], -20, 20,
    #                  color=(0.75, 0.75, 0.75))
    ax2.scatter(df['t'], df['max_v'], color=(0.75, 0.75, 0.75))
    ax2.scatter(df['t'], df['min_v'], color=(0.75, 0.75, 0.75))
    ax2.set_ylim(-15, 15)
    ax2.set_xlabel('Time (mins)')
    ax2.set_ylabel(r'Max Velocity ($\mu m  / s$)')
    # plt.show()
    

    ax1.scatter(df['t'], df['max_y'], color=(0.75, 0.75, 0.75))
    ax1.scatter(df['t'], df['min_y'], color=(0.75, 0.75, 0.75))
    # ax1.set_ylim(0, 350)
    ax1.set_xlabel('Time (mins)')
    ax1.set_ylabel(r'Max Velocity Location ($\mu m$)')
    # plt.show()
    
    
    # Calculate band spread speeds at initial point and end points
    cable_speed = []
    flow_speed = []
    growth_speed = []
    
    colors = ['C0', 'C1', 'C2', 'C3']
    
    for i, range_ in enumerate(analysis_ranges):
        if range_ is None:
            cable_speed.append(np.nan)
            flow_speed.append(np.nan)
            growth_speed.append(np.nan)
            continue
        
        sub = df[(df['t'] >= range_[0]) & (df['t'] <= range_[1])]
        
        if i % 2 == 0:
            point = 'min_y'
            vel = 'min_v'
            mult = 1
        else:
            point = 'max_y'
            vel = 'max_v'
            mult = -1
        
            
        # Draw over points
        ax2.scatter(sub['t'], sub[vel], color=colors[i])
        ax1.scatter(sub['t'], sub[point], color=colors[i])
        
        # Linear fit to band position
        speed, inter = np.polyfit(sub['t'], sub[point], 1)
        ax1.plot(sub['t'], sub['t'] * speed + inter, color='k', ls='--')
        
        cable_speed.append(speed * mult)
        
        # Mean velocity
        flow_speed.append(sub[vel].mean() * mult)
        
        # Growth speed
        growth_speed.append(cable_speed[-1] - flow_speed[-1])
        
    
    ax3.bar([0, 1, 3, 4],
            flow_speed,
            color=colors)
    # plt.xlabel()
    ax3.axhline(0, lw=0.75, color='k')
    ax3.set_xticks([0.5, 3.5], ['Before', 'After'])
    ax3.set_ylabel('Mean Contractile Velocity ($\mu m$)')
    ax3.set_ylim(-15, 30)
    ax3.set_xlim(-0.5, 4.5)
    # plt.show()
    
    # Mean cable speeds
    ax4.bar([0, 1, 3, 4],
            cable_speed,
            color=colors)
    ax4.axhline(0, lw=0.75, color='k')
    ax4.set_xticks([0.5, 3.5], ['Before', 'After'])
    ax4.set_ylabel('Mean Cable End Velocity ($\mu m$)')
    ax4.set_ylim(-15, 30)
    ax4.set_xlim(-0.5, 4.5)
    # plt.show()
    
    
    # Mean cable growth speeds
    ax5.bar([0, 1, 3, 4],
            growth_speed,
            color=colors)
    ax5.set_xticks([0.5, 3.5], ['Before', 'After'])
    ax5.set_ylabel('Mean Cable Growth Velocity ($\mu m$)')
    ax5.axhline(0, lw=0.75, color='k')
    ax5.set_xlim(-0.5, 4.5)
    plt.ylim(-15, 30)
    
    fig.set_size_inches(15, 8)
    plt.show()
    
    
    # Fit hydrodynamic length scales
    hydro_ls = []
    
    for i, range_ in enumerate(analysis_ranges):
        
        if range_ is None:
            hydro_ls.append(np.nan)
            continue
            
        if i % 2 == 0:
            v_vs_y = v_vs_y_max
        else:
            v_vs_y = v_vs_y_min

        v_vs_y = v_vs_y[(v_vs_y['t'] >= range_[0]) & (v_vs_y['t'] <= range_[1])]

        hydro_ls.append(fit_exponential_decay(v_vs_y['y'], v_vs_y['v']))
    
    
    # Return estimates for use gathering statistics
    return flow_speed, cable_speed, growth_speed, [True, True, False, False], hydro_ls
    

if __name__ == '__main__':


    # Data : [file_path, type, duration, interval, center_x, angle, analysis range, pix interval]
    exps = [['unperturbed/ 20210829_15s/', 'unperturbed', 119, 15, 203.19, 0, [[1, 6], [1, 4], None, None], 16],
            ['unperturbed/20211207_1_15s/', 'unperturbed', 124, 15, 83.85, 0, [[0, 10], None, None, None], 16],
            ['unperturbed/20211207_2_15s/', 'unperturbed', 149, 15, 71.37, 0, [[3, 15], None, None, None], 16],
            ['unperturbed/20230328_15s/', 'unperturbed', 172, 15, 214.89, 0, [[0, 10], None, None, None], 16],
            ['unperturbed/20230329_1/', 'unperturbed', 194, 15, 0, 228, [None, None, None, None], 16],
            ['unperturbed/20230329_1/', 'unperturbed', 184, 15, 0, 195, [None, None, None, None], 16],
            ['unperturbed/20230329_3/', 'unperturbed', 170, 15, -101.01, 90, [None, None, None, None], 16]
            ]
    
    exps += [['SbTub/20220215_20s/', 'SbTub', 89, 20, 215.67, 0, [None, None, None, [27, 35]], 32],
             ['SbTub/20221202_30s/', 'SbTub', 94, 30, -85, 73, [None, None, [21, 28], [21, 28]], 32],
             ['SbTub/20230510_embryo1_15s/', 'SbTub', 153, 15, 188.76, 0, [None, None, [10, 15], None], 32],
             ['SbTub/20230510_embryo2_15s/', 'SbTub', 143, 15, 188.76, 0, [None, None, [20, 25], None], 32],
             ['SbTub/20230516_embryo1_15s/', 'SbTub', 197, 15, 201.24, 0, [None, None, [5, 17], None], 32],
             ['SbTub/20230516_embryo2_15s/', 'SbTub', 179, 15, 176.28, 0, [None, None, [20, 28], None], 32],
             ['SbTub/20230516_embryo3_15s/', 'SbTub', 159, 15, 238.68, 0, [None, None, [8, 15], None], 32],
             ['SbTub/20230516_embryo4_15s/', 'SbTub', 66, 15, 151.32, 0, [None, None, [8, 12], None], 32]
            ]
    
    # Ranges checked! First PIV could be in a bigger window?
    exps += [['CHX/20230314_dcx-utr_CHX_15s/', 'CHX', 199, 15, -201.24, 90, [None, None, None, [35, 50]], 32],
            ['CHX/20230316_dcx-utr_CHX_15s/', 'CHX', 249, 15, 265, -38, [None, None, None, [20, 60]], 32],
            ['CHX/20230428_dcx-utr_CHX_embryo1_20s/', 'CHX', 230, 20, 176.28, 0, [None, None, [10, 35], None], 32],
            ['CHX/20230428_dcx-utr_CHX_embryo2_20s/', 'CHX', 227, 20, 238.68, 0, [None, None, [10, 50], None], 32],
            # ['CHX/20230510_dcx-utr_CHX_15s/', 'CHX', 174, 15, 40, 233, [None, None, None, None], 32],
            ['CHX/20230512_dcx-utr_CHX_15s/', 'CHX', 179, 15, 250, -45, [None, None, [30, 45], None], 32],
            ['CHX/20230517_dcx-utr_CHX_15s_embryo1/', 'CHX', 227, 15, 188.76, 0, [None, None, [10, 25], None], 32],
            ['CHX/20230517_dcx-utr_CHX_15s_embryo2/', 'CHX', 115, 15, 238.68, 0, [None, None, [8, 20], None], 32],
            ['CHX/20230523_dcx-utr_CHX_embryo1_20s/', 'CHX', 233, 20, 0, 40, [None, None, [20, 60], None], 32],
            ['CHX/20230523_dcx-utr_CHX_embryo2_15s/', 'CHX', 296, 15, 176.28, 0, [None, None, [5, 20], None], 32]
            ]
    # Make into data frame format
    data = pd.DataFrame(data=exps,
                        columns=['path',
                                 'type',
                                 'duration',
                                 'interval',
                                 'center',
                                 'angle',
                                 'analysis_range',
                                 'pix_interval'])
    
    

    
    width = 25
    
    # Flow rates
    for i in [14]:
        row=data.iloc[i]
        t = 40
        check_angle(row['path'], t, row['angle'], row['center'])
        
        get_hydro_decay(row['path'],
                        t,
                        row['interval'],
                        row['center'],
                        width,
                        row['angle'])
        
        
    for i in [0]:
        row=data.iloc[i]
        t = 8
        check_angle(row['path'], t, row['angle'], row['center'])
        
        get_hydro_decay(row['path'],
                        t,
                        row['interval'],
                        row['center'],
                        width,
                        row['angle'])
        
        
    for i in [-1]:
        row=data.iloc[i]
        check_angle(row['path'], 20, row['angle'], row['center'])

    
    # Make PIV images
    for i in range(len(data)):
        row = data.iloc[i]
        plot_piv(row['path'], row['duration'], row['interval'], vmax=10)
        
    # Use the following bash command in the image folder to make a movie
    # ffmpeg -framerate 10 -i piv_image_%06d.png -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" piv_image.mp4
    

    
    flow_speed, cable_speed, growth_speed, start, hydros = [], [], [], [], []
    exp_type = []
    # for i in [14]:
    for i in range(len(data)):
        
        row = data.iloc[i]
        print(row['path'])
        flow, cable, growth, s, ls = flow_and_growth_analysis(row['path'],
                                                              row['duration'],
                                                              row['interval'],
                                                              row['center'],
                                                              width,
                                                              row['analysis_range'],
                                                              row['angle'])
    
                             
        flow_speed += flow
        cable_speed += cable
        growth_speed += growth
        start += s
        hydros += ls
        
        exp_type += 4 * [row['type']]
    
    
    df = pd.DataFrame(data={'flow_speed': flow_speed,
                            'cable_speed': cable_speed,
                            'growth_speed': growth_speed,
                            'start': start,
                            'type': exp_type,
                            'hydrodynamic_length': hydros})
    

    df = df[~df['flow_speed'].isna()]
    
    # df.to_csv('piv_analysis.csv')
    
    
    
    
    """
    Analysis
    """
    
    df = pd.read_csv('piv_analysis.csv')
    
    start = df[df['start']]
    end = df[~df['start']]
    depol = end[end['type'] == 'SbTub']
    chx = end[end['type'] == 'CHX']
    
    subs = [start, depol, chx]
    xlabels = ['Unperturbed', 'MT Depol', 'Interphase\nArrest']
    
    grey = (0.75, 0.75, 0.75)
    colors = ['C0', 'C1', 'C2']
    
    variables = ['flow_speed', 'cable_speed', 'growth_speed']
    ylabels = ['Contractile Velocity ($\mu m / min$)',
                'Cable Velocity ($\mu m / min$)',
                'Growth Velocity ($\mu m / min$)']

    for j, var in enumerate(variables):
        # T-tests
        
        
        plt.axhline(0, lw=0.75, color='k')
        for i, sub in enumerate(subs):
            plt.scatter(np.random.uniform(-1/6, 1/6, len(sub)) + i, sub[var], color=grey)
            plt.plot([i - 1/3, i + 1/3], [sub[var].mean()] * 2, color=colors[i])
            plt.plot([i, i], [sub[var].mean() - sub[var].std(), sub[var].mean() + sub[var].std()], color=colors[i])

            _, p = ttest_ind(subs[i][var], subs[(i+1)%3][var])
            print('P value between', xlabels[i], 'and', xlabels[(i+1)%3], p)
            # Draw
            if p > 0.05:
                text = 'ns'
            else:
                # text = '* : $p < 10^{:}$'.format(int(np.log10(p)))
                text = '* : p < 0.05'
                
            x1, x2 = i, (i+1) % 3
            y, h = 40 + 6 * i, 1
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='k')
            plt.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color='k')
            
        plt.xticks([0, 1, 2], labels=xlabels)
        plt.ylabel(ylabels[j])
        plt.xlim(-0.5, 2.5)
        plt.ylim(-20, 60)
        
        plt.show()
        
    
    # Retratction only
    var = 'cable_speed'
    subs = [depol, chx]
    xlabels = ['MT\nDepolymerisation', 'Interphase\nArrest']
    
    colors = [[39, 81, 143],
              [226, 119, 104]]
    
    for col in colors:
        for i in range(3):
            col[i] = col[i] / 255
    
    
    # colors = ['C0', 'C1']

    for i, sub in enumerate(subs):
        plt.scatter(1 + np.random.uniform(-1/8, 1/8, len(sub)) + i, -sub[var], color=grey, zorder=-100)
        plt.plot([1 + i - 1/3, 1 + i + 1/3], [-sub[var].mean()] * 2, color=colors[i], marker='none')
        plt.plot([1 + i, 1 + i], [-sub[var].mean() - sub[var].std(), -sub[var].mean() + sub[var].std()], color=colors[i], marker='none')
        
        # plt.errorbar([1 + i], [-sub[var].mean()], yerr=sub[var].std(), color=colors[i])
        # plt.
        
        
    # # Or boxplot
    # data = [list(-depol[var]), list(-chx[var])]
    # bp = plt.boxplot(data, sym='', widths=0.5, patch_artist=True)
    
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)
    
    # Statistics
    _, p = ttest_ind(subs[0][var], subs[1][var])
    print('P value between', xlabels[0], 'and', xlabels[1], p)
    print('MT Mean', subs[0][var].mean(), '+-', subs[0][var].std())
    print('SbTub Mean', subs[1][var].mean(), '+-', subs[1][var].std())
    # Draw
    if p > 0.01:
        text = 'ns'
    else:
        # text = '* : $p < 10^{:}$'.format(int(np.log10(p)))
        text = '* : p < 0.01'
        

    x1, x2 = 1 + i, 1 + (i+1) % 2
    y, h = 40, 1
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='k', marker='none')
    plt.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color='k')
        
    plt.xticks([1, 2], labels=xlabels)
    plt.ylabel('Retraction Speed ($\mu m / min$)')
    plt.ylim(0, 45)
    plt.xlim(0.5, 2.5)
    # plt.gcf().set_size_inches(4, 4)
    plt.gcf().savefig('sbtub vs chx retraction speed.svg', format='svg', bbox_inches='tight')
    plt.show()
