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
from functools import partial

def matrices_cluster(grupo, evocados_co, evocados_in):
    # Diccionario con evocados de todos los sujetos, para ensayos congruentes e incongruentes
    congruentes = copy.deepcopy(evocados_co)
    incongruentes = copy.deepcopy(evocados_in)
    tiempos = congruentes['22100'].times
    adjacency, canales = find_ch_adjacency(congruentes['22100'].info, "eeg")
    
    # Matrices de evocados (tiempos, canales) de sujetos condición 1
    grupo_co = {}
    grupo_in = {}
    for sujeto in grupo:
        grupo_co[sujeto] = congruentes[sujeto]
        grupo_in[sujeto] = incongruentes[sujeto]

    ga_grupo_co = mne.grand_average(list(grupo_co.values()), interpolate_bads=True, drop_bads=True)
    ga_grupo_in = mne.grand_average(list(grupo_in.values()), interpolate_bads=True, drop_bads=True)
    
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
    df.set_index('Time', inplace=True)
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

# Función que obtiene los interavlos de tiempo en los que se presentan los clústeres
def time_intervals(df, electrodos):
    df1 = df.copy()
    df1.reset_index(inplace=True)
    df1['match'] = True
    for i in range(0,len(electrodos)):
        df1['match'] = (df1[electrodos[i]]==1)&(df1['match'])
    df1['match'] = df1['match'].astype(int)
    df1['cuenta'] = 0
    df1['mayor_racha'] = 0
    for i in range(0,len(df1)):
        if df1.at[i,'match'] == 1:
            df1.at[i,'cuenta'] = 1 + df1.at[i-1,'cuenta']
        else:
            df1.at[i,'cuenta'] = 0
    for i in range(0,len(df1)-1):
        if (df1.at[i,'cuenta'] > 11) & (df1.at[i+1,'cuenta']==0):
            df1.at[i,'mayor_racha'] = 1
    if df1.at[len(df1)-1,'cuenta'] > 11:
        df1.at[len(df1)-1,'mayor_racha'] = 1
    df1['t_inicial'] = 0.
    for i in df1.index:
        if df1.at[i,'mayor_racha']==1:
            df1.at[i,'t_inicial'] = df1.at[i +1 - df1.at[i,'cuenta'],'Time']
    intervals = df1[df1['mayor_racha']==1][['cuenta','t_inicial','Time']].copy()
    intervals.rename(columns={'Time':'t_final'}, inplace=True)
    intervals['delta(ms)'] = (intervals['t_final'] - intervals['t_inicial'])*1000
    return intervals