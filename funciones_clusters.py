import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import re
import pandas as pd
import seaborn as sns
import copy
from mne.viz import plot_evoked_topo
from mne.channels import find_ch_adjacency, make_1020_channel_selections
from mne.stats import spatio_temporal_cluster_test, ttest_ind_no_p, spatio_temporal_cluster_1samp_test
from functools import partial, reduce

def matrices_cluster(grupo, evocados_co, evocados_in, tmin=0, tmax=None):
    # Diccionario con evocados de todos los sujetos, para ensayos congruentes e incongruentes
    congruentes = copy.deepcopy(evocados_co)
    incongruentes = copy.deepcopy(evocados_in)
    adjacency, canales = find_ch_adjacency(congruentes['22100'].info, "eeg")

    # Matrices de evocados (tiempos, canales) de sujetos condición 1
    grupo_co = {}
    grupo_in = {}
    for sujeto in grupo:
        grupo_co[sujeto] = congruentes[sujeto].crop(tmin=tmin, tmax=tmax)
        grupo_in[sujeto] = incongruentes[sujeto].crop(tmin=tmin, tmax=tmax)

    
    ga_grupo_co = mne.grand_average(list(grupo_co.values()), interpolate_bads=True, drop_bads=True)
    ga_grupo_in = mne.grand_average(list(grupo_in.values()), interpolate_bads=True, drop_bads=True)

    sujeto = list(grupo_co.keys())[0]
    tiempos = grupo_co[sujeto].times
    co_grupo = [grupo_co[i].data for i in grupo_co]
    co_grupo = np.reshape(co_grupo,(-1,len(canales), len(tiempos)))
    co_grupo = np.transpose(co_grupo, axes=(0,2,1))
    print(co_grupo.shape)

    in_grupo = [grupo_in[i].data for i in grupo_in]
    in_grupo = np.reshape(in_grupo,(-1,len(canales), len(tiempos)))
    in_grupo = np.transpose(in_grupo, axes=(0,2,1))
    print(in_grupo.shape)

    return (ga_grupo_co, ga_grupo_in, co_grupo, in_grupo, tiempos, canales, adjacency)

# Análisis por permutaciones usando la prueba t
def analisis_cluster_t(X, adjacency):
    stat_fun_hat = partial(ttest_ind_no_p, sigma=1e-3)
    tfce = dict(start=.2, step=.2)
    T_obs, ___, p_values, ___ = \
    spatio_temporal_cluster_test(X, threshold=tfce, stat_fun=stat_fun_hat,
                                 n_permutations=1000, tail=1,adjacency=adjacency, n_jobs=1, seed=1, out_type='indices',
                                 buffer_size=None)

    p_values = p_values.reshape(T_obs.shape)
    significant_points = p_values.T < .05
    print(str(significant_points.sum()) + " points selected by TFCE ...")
    try:
        idx = np.where(p_values <= 0.05)
        umbral = T_obs[idx].min()
    except:
        print('No hay clústeres con p-value <= 0.05')
        umbral = 0
    return (T_obs, p_values, umbral)

# Análisis por permutaciones usando ANOVA
def analisis_cluster_f(X, adjacency):
    tfce = dict(start=.2, step=.2)
    T_obs, ___, p_values, ___ = \
    spatio_temporal_cluster_test(X, threshold=tfce, n_permutations=1000,
                                 tail=1,adjacency=adjacency, n_jobs=1, seed=1, out_type='indices',
                                 buffer_size=None)

    p_values = p_values.reshape(T_obs.shape)
    significant_points = p_values.T < .05
    print(str(significant_points.sum()) + " points selected by TFCE ...")
    try:
        idx = np.where(p_values <= 0.05)
        umbral = T_obs[idx].min()
    except:
        print('No hay clústeres con p-value <= 0.05')
        umbral = 0
    return (T_obs, p_values, umbral)

# Análisis por permutaciones una sola muestra
def analisis_cluster_paired(X, adjacency):
    tfce = dict(start=.2, step=.2)
    T_obs, ___, p_values, ___ = \
    spatio_temporal_cluster_1samp_test(X, threshold=tfce, n_permutations=1000,
                                       tail=1,adjacency=adjacency, n_jobs=1, seed=1,
                                       out_type='indices', buffer_size=None)

    p_values = p_values.reshape(T_obs.shape)
    significant_points = p_values.T < .05
    print(str(significant_points.sum()) + " points selected by TFCE ...")
    try:
        idx = np.where(p_values <= 0.05)
        umbral = T_obs[idx].min()
    except:
        print('No hay clústeres con p-value <= 0.05')
        umbral = 0
    return (T_obs, p_values, umbral)

# Gráficas de clústeres
def clusters_color(df, umbral=0.5, x_labels=8, cmap='hot', cbar=True):
    df = df[df['Time']>= 0]
    df['Time'] = ((df['Time'].round(3))*1000).astype(int)
    df.rename(columns={'Time':'Time (ms)'}, inplace=True)
    df.set_index('Time (ms)', inplace=True)
    df = df.transpose()
    mask = df>umbral
    # cmap = sns.diverging_palette(h_neg=0 , h_pos=240, s=50, l=50, sep=1, n=6, center='light', as_cmap=False)
    figura = sns.heatmap(df, vmin=0.0, vmax=umbral, cmap=cmap, linewidths=.05, linecolor='white', xticklabels=x_labels, yticklabels=4, square=False, mask=mask, cbar=cbar)
    plt.show()
    return (figura)

# Función que entrega tabla de valores p ya lista para graficar
def p_values_table(p_values, tiempos, canales):
    df_p_values = pd.DataFrame.from_records(p_values, index=tiempos, columns=canales)
    df_p_values.reset_index(inplace=True)
    df_p_values.rename(columns = {'index': 'Time'}, inplace=True)
    return df_p_values

# Función que entrega listado de clústeres ordenados por tiempo de inicio
def lista_clusteres(df_p_values, umbral=0.5):
    df = df_p_values.copy()
    df.set_index('Time', inplace=True)
    column_dict = {}
    for col in [{col: {True: col}} for col in df.columns]:
        column_dict.update(col)
    df = (df <= umbral).replace(to_replace=column_dict)
    lista = [tuple(y for y in x if y != False) for x in df.to_records()]
    clusteres = []
    for tupla in range(len(lista)):
        if len(lista[tupla]) > 1:
            clusteres.append(lista[tupla])
    time_point = []
    electrodes = []
    for cluster in range(len(clusteres)):
        time_point.append(clusteres[cluster][0])
        electrodes.append(clusteres[cluster][1:])
    clusteres_dict = {'time_point': time_point, 'electrodes': electrodes}
    df_clusteres = pd.DataFrame(clusteres_dict)
    return df_clusteres

# Función que obtiene los electrodos que hacen parte de algún clúster
def lista_electrodos(df_p_values, umbral):
    df = df_p_values.copy()
    df.set_index('Time', inplace=True)
    for idx in df.index:
        for col in df.columns:
            if (df.loc[idx, col] <= umbral):
                df.at[idx, col] = 1
            else:
                df.at[idx, col] = 0
    lista = df.sum(axis=0)
    lista = lista.sort_values(ascending=False)
    return lista, df

# Función que obtiene los intervalos de tiempo en los que se presentan los clústeres
def time_intervals(df, electrodos, min_duration=40):
    min_muestras = np.round(min_duration*.256, decimals=0)+1
    df1 = df.copy()
    df1.reset_index(inplace=True)
    df1['match'] = True
    for i in range(0,len(electrodos)):
        df1['match'] = (df1[electrodos[i]]==1)&(df1['match'])
    df1['match'] = df1['match'].astype(int)
    df1['cuenta'] = 0
    df1['mayor_racha'] = 0
    if df1.at[0,'match'] == 1:
        df1.at[0,'cuenta'] = 1
    for i in range(1,len(df1)):
        if df1.at[i,'match'] == 1:
            df1.at[i,'cuenta'] = 1 + df1.at[i-1,'cuenta']
        else:
            df1.at[i,'cuenta'] = 0
    for i in range(0,len(df1)-1):
        if (df1.at[i,'cuenta'] > min_muestras) & (df1.at[i+1,'cuenta']==0):
            df1.at[i,'mayor_racha'] = 1
    if df1.at[len(df1)-1,'cuenta'] > min_muestras:
        df1.at[len(df1)-1,'mayor_racha'] = 1
    df1['t_inicial'] = 0.
    for i in df1.index:
        if df1.at[i,'mayor_racha']==1:
            df1.at[i,'t_inicial'] = df1.at[i +1 - df1.at[i,'cuenta'],'Time']
    intervals = df1[df1['mayor_racha']==1][['cuenta','t_inicial','Time']].copy()
    intervals.rename(columns={'Time':'t_final'}, inplace=True)
    intervals['delta(ms)'] = (intervals['t_final'] - intervals['t_inicial'])*1000
    return intervals

# Evocados de clústeres
def ROI_evoked(evocado, canales):
    lista_canales = evocado.ch_names
    lista_indices = []
    for canal in canales:
        indice = lista_canales.index(canal)
        lista_indices.append(indice)
    ROI_evocado = mne.channels.combine_channels(evocado, groups={'ROI':lista_indices})
    return ROI_evocado

# Cálculo de media de clúster
def medidas(evoked, tmin, tmax):
    evoked_ROI = evoked.copy()
    evoked_ROI.crop(tmin = tmin, tmax = tmax)
    data = evoked_ROI.to_data_frame(time_format = None)
    data.set_index('time',inplace=True)
    mean_ROI = data.mean()
    return mean_ROI

# Función que entrega lista de ERPS por grupo y tipo de ensayo, y diccionarios de ERPs por tipo de ensayo
def erps_grupos(canales, evocados_co, evocados_in, exbothsides, exparamilitar, exguerrilla, victim, control, tmin=0.0, tmax=0.796875):
    exbothsides_co = {}
    exbothsides_in = {}
    exbothsides_dif = {}
    for sujeto in exbothsides:
        exbothsides_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        exbothsides_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = exbothsides_co[sujeto].data - exbothsides_in[sujeto].data
        exbothsides_dif[sujeto] = mne.EvokedArray(data=data_dif, info=exbothsides_in[sujeto].info, tmin=exbothsides_in[sujeto].tmin)
    victim_co = {}
    victim_in = {}
    victim_dif = {}
    for sujeto in victim:
        victim_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        victim_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = victim_co[sujeto].data - victim_in[sujeto].data
        victim_dif[sujeto] = mne.EvokedArray(data=data_dif, info=victim_in[sujeto].info, tmin=victim_in[sujeto].tmin)
    exparamilitar_co = {}
    exparamilitar_in = {}
    exparamilitar_dif = {}
    for sujeto in exparamilitar:
        exparamilitar_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        exparamilitar_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = exparamilitar_co[sujeto].data - exparamilitar_in[sujeto].data
        exparamilitar_dif[sujeto] = mne.EvokedArray(data=data_dif, info=exparamilitar_in[sujeto].info, tmin=exparamilitar_in[sujeto].tmin)
    exguerrilla_co = {}
    exguerrilla_in = {}
    exguerrilla_dif = {}
    for sujeto in exguerrilla:
        exguerrilla_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        exguerrilla_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = exguerrilla_co[sujeto].data - exguerrilla_in[sujeto].data
        exguerrilla_dif[sujeto] = mne.EvokedArray(data=data_dif, info=exguerrilla_in[sujeto].info, tmin=exguerrilla_in[sujeto].tmin)
    control_co = {}
    control_in = {}
    control_dif = {}
    for sujeto in control:
        control_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        control_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = control_co[sujeto].data - control_in[sujeto].data
        control_dif[sujeto] = mne.EvokedArray(data=data_dif, info=control_in[sujeto].info, tmin=control_in[sujeto].tmin)
    ROI_co = {**exbothsides_co, **victim_co, **exparamilitar_co, **exguerrilla_co, **control_co}
    ROI_in = {**exbothsides_in, **victim_in, **exparamilitar_in, **exguerrilla_in, **control_in}
    ROI_dif = {**exbothsides_dif, **victim_dif, **exparamilitar_dif, **exguerrilla_dif, **control_dif}
    return exbothsides_co, exbothsides_in, exbothsides_dif, victim_co, victim_in, victim_dif, exparamilitar_co, exparamilitar_in, exparamilitar_dif, exguerrilla_co, exguerrilla_in, exguerrilla_dif, control_co, control_in, control_dif, ROI_co, ROI_in, ROI_dif

def graficos_grupos(canales, evocados_co, evocados_in, exbothsides, exparamilitar, exguerrilla, victim, control, t_init, duration, tmin=0.0, tmax=0.796875, pos_legend='lower left'):
    __, __, __, victim_co, victim_in, __, exparamilitar_co, exparamilitar_in, __, exguerrilla_co, exguerrilla_in, __, __, __, __, __, __, __ = erps_grupos(canales, evocados_co, evocados_in, exbothsides, exparamilitar, exguerrilla, victim, control, tmin=0.0, tmax=0.796875)
    victim_co_list = list(victim_co.values())
    victim_in_list = list(victim_in.values())
    exparamilitar_co_list = list(exparamilitar_co.values())
    exparamilitar_in_list = list(exparamilitar_in.values())
    exguerrilla_co_list = list(exguerrilla_co.values())
    exguerrilla_in_list = list(exguerrilla_in.values())
    erps = {'exparamilitar_co':exparamilitar_co_list, 'exparamilitar_in':exparamilitar_in_list, 'exguerrilla_co':exguerrilla_co_list, 'exguerrilla_in':exguerrilla_in_list, 'victim_co':victim_co_list, 'victim_in':victim_in_list}
    fig = mne.viz.plot_compare_evokeds(evokeds=erps, colors=['#1f77b4ff', '#ff7f0eff', '#2ca02cff', '#d62728ff', '#9467bdff', '#8c564bff'], 
                                        linestyles=['solid', 'solid','dashdot', 'dashdot', 'dashed', 'dashed'], axes=None, ci=None, truncate_yaxis=False, 
                                        truncate_xaxis=False, show_sensors=False, legend=pos_legend, split_legend=False, combine=None, show=False)
    ax = fig[0].gca()
    rect = plt.Rectangle((t_init,-1.8), duration, 3.6, color='lightcyan')
    ax.add_patch(rect)
    return ax

def graficos_grupos_control(canales, evocados_co, evocados_in, exbothsides, exparamilitar, exguerrilla, victim, control, t_init, duration, tmin=0.0, tmax=0.796875, pos_legend='lower left'):
    __, __, __, victim_co, victim_in, __, exparamilitar_co, exparamilitar_in, __, exguerrilla_co, exguerrilla_in, __, control_co, control_in, __, __, __, __ = erps_grupos(canales, evocados_co, evocados_in, exbothsides, exparamilitar, exguerrilla, victim, control, tmin=0.0, tmax=0.796875)
    victim_co_list = list(victim_co.values())
    victim_in_list = list(victim_in.values())
    exparamilitar_co_list = list(exparamilitar_co.values())
    exparamilitar_in_list = list(exparamilitar_in.values())
    exguerrilla_co_list = list(exguerrilla_co.values())
    exguerrilla_in_list = list(exguerrilla_in.values())
    control_co_list = list(control_co.values())
    control_in_list = list(control_in.values())
    erps = {'exparamilitar_co':exparamilitar_co_list, 'exparamilitar_in':exparamilitar_in_list, 
            'exguerrilla_co':exguerrilla_co_list, 'exguerrilla_in':exguerrilla_in_list, 
            'victim_co':victim_co_list, 'victim_in':victim_in_list, 
            'control_co':control_co_list, 'control_in':control_in_list}
    fig = mne.viz.plot_compare_evokeds(evokeds=erps, 
                                       linestyles=['solid', 'solid','dashdot', 'dashdot', 'dashed', 'dashed', 'dotted', 'dotted'], 
                                       axes=None, ci=None, truncate_yaxis=False, truncate_xaxis=False, show_sensors=False, 
                                       legend=pos_legend, split_legend=False, combine=None, show=False)
    ax = fig[0].gca()
    rect = plt.Rectangle((t_init,-1.8), duration, 3.6, color='lightcyan')
    ax.add_patch(rect)
    return ax

# Función que entrega lista de ERPS por grupo y tipo de ensayo, y diccionarios de ERPs por tipo de ensayo
def erps_exc_vic(canales, evocados_co, evocados_in, excombatant, victim, control, tmin=0.0, tmax=0.796875):
    victim_co = {}
    victim_in = {}
    victim_dif = {}
    for sujeto in victim:
        victim_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        victim_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = victim_co[sujeto].data - victim_in[sujeto].data
        victim_dif[sujeto] = mne.EvokedArray(data=data_dif, info=victim_in[sujeto].info, tmin=victim_in[sujeto].tmin)
    excombatant_co = {}
    excombatant_in = {}
    excombatant_dif = {}
    for sujeto in excombatant:
        excombatant_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        excombatant_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = excombatant_co[sujeto].data - excombatant_in[sujeto].data
        excombatant_dif[sujeto] = mne.EvokedArray(data=data_dif, info=excombatant_in[sujeto].info, tmin=excombatant_in[sujeto].tmin)
    control_co = {}
    control_in = {}
    control_dif = {}
    for sujeto in control:
        control_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        control_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = control_co[sujeto].data - control_in[sujeto].data
        control_dif[sujeto] = mne.EvokedArray(data=data_dif, info=control_in[sujeto].info, tmin=control_in[sujeto].tmin)
    ROI_co = {**victim_co, **excombatant_co, **control_co}
    ROI_in = {**victim_in, **excombatant_in, **control_in}
    ROI_dif = {**victim_dif, **excombatant_dif, **control_dif}
    return victim_co, victim_in, victim_dif, excombatant_co, excombatant_in, excombatant_dif, control_co, control_in, control_dif, ROI_co, ROI_in, ROI_dif

def graficos_exc_vic(canales, evocados_co, evocados_in, excombatant, victim, control, t_init, duration, tmin=0.0, tmax=0.796875, pos_legend='lower left'):
    victim_co, victim_in, __, excombatant_co, excombatant_in, __, __, __, __, __, __, __ = erps_exc_vic(canales, evocados_co, evocados_in, excombatant, victim, control, tmin=0.0, tmax=0.796875)
    victim_co_list = list(victim_co.values())
    victim_in_list = list(victim_in.values())
    excombatant_co_list = list(excombatant_co.values())
    excombatant_in_list = list(excombatant_in.values())
    erps = {'excombatant_co':excombatant_co_list, 'excombatant_in':excombatant_in_list, 'victim_co':victim_co_list, 'victim_in':victim_in_list}
    fig = mne.viz.plot_compare_evokeds(evokeds=erps, colors=['#1f77b4ff', '#ff7f0eff', '#2ca02cff', '#d62728ff'], 
                                        linestyles=['solid', 'solid','dashdot', 'dashdot'], axes=None, ci=None, truncate_yaxis=False, 
                                        truncate_xaxis=False, show_sensors=False, legend=pos_legend, split_legend=False, combine=None, show=False)
    ax = fig[0].gca()
    rect = plt.Rectangle((t_init,-1.8), duration, 3.6, color='lightcyan')
    ax.add_patch(rect)
    return ax

def erps_self_vic(canales, evocados_co, evocados_in, no_victim, victim, tmin=0.0, tmax=0.796875):
    victim_co = {}
    victim_in = {}
    victim_dif = {}
    for sujeto in victim:
        victim_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        victim_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = victim_co[sujeto].data - victim_in[sujeto].data
        victim_dif[sujeto] = mne.EvokedArray(data=data_dif, info=victim_in[sujeto].info, tmin=victim_in[sujeto].tmin)
    no_victim_co = {}
    no_victim_in = {}
    no_victim_dif = {}
    for sujeto in no_victim:
        no_victim_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        no_victim_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = no_victim_co[sujeto].data - no_victim_in[sujeto].data
        no_victim_dif[sujeto] = mne.EvokedArray(data=data_dif, info=no_victim_in[sujeto].info, tmin=no_victim_in[sujeto].tmin)
    ROI_co = {**victim_co, **no_victim_co}
    ROI_in = {**victim_in, **no_victim_in}
    ROI_dif = {**victim_dif, **no_victim_dif}
    return victim_co, victim_in, victim_dif, no_victim_co, no_victim_in, no_victim_dif, ROI_co, ROI_in, ROI_dif

def graficos_self_vic(canales, evocados_co, evocados_in, no_victim, victim, t_init, duration, tmin=0.0, tmax=0.796875, pos_legend='lower left'):
    victim_co, victim_in, __, no_victim_co, no_victim_in, __, __, __, __ = erps_self_vic(canales, evocados_co, evocados_in, no_victim=no_victim, victim=victim, tmin=0.0, tmax=0.796875)
    victim_co_list = list(victim_co.values())
    victim_in_list = list(victim_in.values())
    no_victim_co_list = list(no_victim_co.values())
    no_victim_in_list = list(no_victim_in.values())
    erps = {'no_victim_co':no_victim_co_list, 'no_victim_in':no_victim_in_list, 'victim_co':victim_co_list, 'victim_in':victim_in_list}
    fig = mne.viz.plot_compare_evokeds(evokeds=erps, colors=['#1f77b4ff', '#ff7f0eff', '#2ca02cff', '#d62728ff'], 
                                        linestyles=['solid', 'solid','dashdot', 'dashdot'], axes=None, ci=None, truncate_yaxis=False, 
                                        truncate_xaxis=False, show_sensors=False, legend=pos_legend, split_legend=False, combine=None, show=False)
    ax = fig[0].gca()
    rect = plt.Rectangle((t_init,-1.8), duration, 3.6, color='lightcyan')
    ax.add_patch(rect)
    return ax

# Función que entrega lista de ERPs de clústeres por nivel y tipo de ensayo, y diccionarios de ERPs por tipo de ensayo
def erps_niveles(canales, evocados_co, evocados_in, positive, negative, neutral, tmin=0.0, tmax=0.796875):
    negative_co = {}
    negative_in = {}
    negative_dif = {}
    for sujeto in negative:
        negative_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        negative_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = negative_co[sujeto].data - negative_in[sujeto].data
        negative_dif[sujeto] = mne.EvokedArray(data=data_dif, info=negative_in[sujeto].info, tmin=negative_in[sujeto].tmin)
    neutral_co = {}
    neutral_in = {}
    neutral_dif = {}
    for sujeto in neutral:
        neutral_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        neutral_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = neutral_co[sujeto].data - neutral_in[sujeto].data
        neutral_dif[sujeto] = mne.EvokedArray(data=data_dif, info=neutral_in[sujeto].info, tmin=neutral_in[sujeto].tmin)
    positive_co = {}
    positive_in = {}
    positive_dif = {}
    for sujeto in positive:
        positive_co[sujeto] = ROI_evoked(evocados_co[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        positive_in[sujeto] = ROI_evoked(evocados_in[sujeto],canales).crop(tmin=tmin,tmax=tmax)
        data_dif = positive_co[sujeto].data - positive_in[sujeto].data
        positive_dif[sujeto] = mne.EvokedArray(data=data_dif, info=positive_in[sujeto].info, tmin=positive_in[sujeto].tmin)
    # Diccionarios de toda la muestra
    ROI_co = {**negative_co,**neutral_co,**positive_co}
    ROI_in = {**negative_in,**neutral_in,**positive_in}
    ROI_dif = {**negative_dif,**neutral_dif,**positive_dif}
    return negative_co, neutral_co, positive_co, negative_in, neutral_in, positive_in,negative_dif,neutral_dif,positive_dif, ROI_co, ROI_in, ROI_dif

# Función que entrega gráficos de ERPs de clústeres por nivel y tipo de ensayo
def graficos_niveles(negative_co, neutral_co, positive_co, negative_in, neutral_in, positive_in,negative_dif,neutral_dif,positive_dif):
    # Configuro parámetros de gráficos
    params = {'axes.labelsize':20, 'axes.titlesize': 20, 'axes.grid':True, 'axes.grid.axis':'both', 'axes.grid.which':'both','legend.fontsize':20, 'legend.loc': 'best', 'lines.linewidth': 2.0 ,'xtick.labelsize': 15, 'xtick.minor.visible': True, 'ytick.labelsize': 15, 'ytick.minor.visible': True}
    plt.rcParams.update(params)
    # Listas de evocados de clústeres por grupos
    negative_co_list = list(negative_co.values())
    neutral_co_list = list(neutral_co.values())
    positive_co_list = list(positive_co.values())
    negative_in_list = list(negative_in.values())
    neutral_in_list = list(neutral_in.values())
    positive_in_list = list(positive_in.values())
    negative_dif_list = list(negative_dif.values())
    neutral_dif_list = list(neutral_dif.values())
    positive_dif_list = list(positive_dif.values())
    erps = {'negative_co':negative_co_list, 'neutral_co':neutral_co_list, 'positive_co':positive_co_list, 'negative_in':negative_in_list, 'neutral_in':neutral_in_list, 'positive_in':positive_in_list}
    # Gráfico de ensayos congruentes e incongruentes, con o sin intervalos de confianza
    figura_1 = mne.viz.plot_compare_evokeds(evokeds=erps, colors=['#1f77b4ff', '#ff7f0eff', '#2ca02cff', '#d62728ff', '#9467bdff', '#8c564bff'], linestyles=['solid', 'dashdot', 'dashed', 'solid', 'dashdot', 'dashed'], axes=None, ci=None, truncate_yaxis=False, truncate_xaxis=False, show_sensors=False, split_legend=False, combine=None, show=False)
    # Gráfico de diferencias intervalos de confianza
    erps_dif = {'negative_dif':negative_dif_list, 'neutral_dif':neutral_dif_list, 'positive_dif':positive_dif_list}
    figura_2 = mne.viz.plot_compare_evokeds(evokeds=erps_dif, colors=['#1f77b4ff', '#ff7f0eff', '#2ca02cff'], linestyles=['solid', 'dashdot', 'dashed'],axes=None, ci=0.95, truncate_yaxis=False, truncate_xaxis=False, show_sensors=False, legend=True, split_legend=False, combine=None, show=False)
    return figura_1, figura_2

def batch_medidas_grupos(clusteres, evocados_co, evocados_in, exbothsides, exparamilitar, exguerrilla, victim, control):
    medidas_clusteres = {}
    for i in clusteres.index:
        canales = clusteres['electrodos'][i].split(', ')
        tmin = clusteres['t_inicial (s)'][i]
        tmax = clusteres['t_final (s)'][i]
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,ROI_co, ROI_in, ROI_dif = erps_grupos(canales=canales, evocados_co=evocados_co, evocados_in=evocados_in, exbothsides=exbothsides, exparamilitar=exparamilitar, exguerrilla=exguerrilla, victim=victim, control=control)
        # Medidas clúster
        medidas_clusteres[i] = pd.DataFrame(columns= ['subject','mean_co', 'mean_in', 'dif_co_in'])
        for k in ROI_co.keys():
            mean_ROI_co = medidas(ROI_co[k], tmin=tmin, tmax=tmax)
            mean_ROI_in = medidas(ROI_in[k], tmin=tmin, tmax=tmax)
            mean_ROI_dif = medidas(ROI_dif[k], tmin=tmin, tmax=tmax)
            medidas_clusteres[i] =  medidas_clusteres[i].append({'subject': k, 'mean_co': mean_ROI_co[0], 'mean_in': mean_ROI_in[0], 'dif_co_in': mean_ROI_dif[0]}, ignore_index=True)
        # Construcción dataframe de medidas de todos los clústeres
    for k in medidas_clusteres.keys():
        num_cluster = str(k)
        medidas_clusteres[k].rename(columns={'mean_co':'mean_co_'+num_cluster,'mean_in':'mean_in_'+num_cluster,'dif_co_in':'dif_co_in_'+num_cluster}, inplace=True)
    df_medidas = reduce(lambda  left,right: pd.merge(left,right,on=['subject'],how='inner'),medidas_clusteres.values())
    return df_medidas

def batch_medidas_exc_vic(clusteres, evocados_co, evocados_in, excombatant, victim, control):
    medidas_clusteres = {}
    for i in clusteres.index:
        canales = clusteres['electrodos'][i].split(', ')
        tmin = clusteres['t_inicial (s)'][i]
        tmax = clusteres['t_final (s)'][i]
        __, __, __, __, __, __, __, __, __,ROI_co, ROI_in, ROI_dif = erps_exc_vic(canales=canales, evocados_co=evocados_co, evocados_in=evocados_in, excombatant=excombatant, victim=victim, control=control)
        # Medidas clúster
        medidas_clusteres[i] = pd.DataFrame(columns= ['subject','mean_co', 'mean_in', 'dif_co_in'])
        for k in ROI_co.keys():
            mean_ROI_co = medidas(ROI_co[k], tmin=tmin, tmax=tmax)
            mean_ROI_in = medidas(ROI_in[k], tmin=tmin, tmax=tmax)
            mean_ROI_dif = medidas(ROI_dif[k], tmin=tmin, tmax=tmax)
            medidas_clusteres[i] =  medidas_clusteres[i].append({'subject': k, 'mean_co': mean_ROI_co[0], 'mean_in': mean_ROI_in[0], 'dif_co_in': mean_ROI_dif[0]}, ignore_index=True)
        # Construcción dataframe de medidas de todos los clústeres
    for k in medidas_clusteres.keys():
        num_cluster = str(k)
        medidas_clusteres[k].rename(columns={'mean_co':'mean_co_'+num_cluster,'mean_in':'mean_in_'+num_cluster,'dif_co_in':'dif_co_in_'+num_cluster}, inplace=True)
    df_medidas = reduce(lambda  left,right: pd.merge(left,right,on=['subject'],how='inner'),medidas_clusteres.values())
    return df_medidas

def batch_medidas_self_vic(clusteres, evocados_co, evocados_in, no_victim, victim):
    medidas_clusteres = {}
    for i in clusteres.index:
        canales = clusteres['electrodos'][i].split(', ')
        tmin = clusteres['t_inicial (s)'][i]
        tmax = clusteres['t_final (s)'][i]
        __, __, __, __, __, __, ROI_co, ROI_in, ROI_dif = erps_self_vic(canales=canales, evocados_co=evocados_co, evocados_in=evocados_in, no_victim=no_victim, victim=victim)
        # Medidas clúster
        medidas_clusteres[i] = pd.DataFrame(columns= ['subject','mean_co', 'mean_in', 'dif_co_in'])
        for k in ROI_co.keys():
            mean_ROI_co = medidas(ROI_co[k], tmin=tmin, tmax=tmax)
            mean_ROI_in = medidas(ROI_in[k], tmin=tmin, tmax=tmax)
            mean_ROI_dif = medidas(ROI_dif[k], tmin=tmin, tmax=tmax)
            medidas_clusteres[i] =  medidas_clusteres[i].append({'subject': k, 'mean_co': mean_ROI_co[0], 'mean_in': mean_ROI_in[0], 'dif_co_in': mean_ROI_dif[0]}, ignore_index=True)
        # Construcción dataframe de medidas de todos los clústeres
    for k in medidas_clusteres.keys():
        num_cluster = str(k)
        medidas_clusteres[k].rename(columns={'mean_co':'mean_co_'+num_cluster,'mean_in':'mean_in_'+num_cluster,'dif_co_in':'dif_co_in_'+num_cluster}, inplace=True)
    df_medidas = reduce(lambda  left,right: pd.merge(left,right,on=['subject'],how='inner'),medidas_clusteres.values())
    return df_medidas

def batch_medidas_niveles(clusteres, evocados_co, evocados_in, positive, negative, neutral):
    medidas_clusteres = {}
    for i in clusteres.index:
        canales = clusteres['electrodos'][i].split(', ')
        tmin = clusteres['t_inicial (s)'][i]
        tmax = clusteres['t_final (s)'][i]
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, ROI_co, ROI_in, ROI_dif = erps_niveles(canales=canales, evocados_co=evocados_co, evocados_in=evocados_in, positive=positive, negative=negative, neutral=neutral)
        # Medidas clúster
        medidas_clusteres[i] = pd.DataFrame(columns= ['subject','mean_co', 'mean_in', 'dif_co_in'])
        for k in ROI_co.keys():
            mean_ROI_co = medidas(ROI_co[k], tmin=tmin, tmax=tmax)
            mean_ROI_in = medidas(ROI_in[k], tmin=tmin, tmax=tmax)
            mean_ROI_dif = medidas(ROI_dif[k], tmin=tmin, tmax=tmax)
            medidas_clusteres[i] =  medidas_clusteres[i].append({'subject': k, 'mean_co': mean_ROI_co[0], 'mean_in': mean_ROI_in[0], 'dif_co_in': mean_ROI_dif[0]}, ignore_index=True)
        # Construcción dataframe de medidas de todos los clústeres
    for k in medidas_clusteres.keys():
        num_cluster = str(k)
        medidas_clusteres[k].rename(columns={'mean_co':'mean_co_'+num_cluster,'mean_in':'mean_in_'+num_cluster,'dif_co_in':'dif_co_in_'+num_cluster}, inplace=True)
    df_medidas = reduce(lambda  left,right: pd.merge(left,right,on=['subject'],how='inner'),medidas_clusteres.values())
    return df_medidas