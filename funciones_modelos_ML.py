import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.inspection import permutation_importance
from mango import Tuner, scheduler
from scipy.stats import uniform
from xgboost import XGBClassifier

# Ajuste modelos de clustering
def clusters_kmeans(data, max_clusters=10):
    inertias = []
    sil_scores = []
    cal_scores = []
    dav_scores = []
    for i in range(2,max_clusters+1):
        kmeans = KMeans(i, random_state=72).fit(data)
        inertia = kmeans.inertia_
        inertias.append(inertia)
        labels = kmeans.labels_
        sil = silhouette_score(X=data, labels=labels)
        sil_scores.append(sil)
        cal = calinski_harabasz_score(X=data, labels=labels)
        cal_scores.append(cal)
        dav = davies_bouldin_score(X=data, labels=labels)
        dav_scores.append(dav)
    return inertias, sil_scores, cal_scores, dav_scores

def clusters_gaussian(data, max_clusters=10):
    gauss_scores = []
    sil_scores = []
    cal_scores = []
    dav_scores = []
    for i in range(2,max_clusters+1):
        gauss = GaussianMixture(i, random_state=72).fit(data)
        labels = gauss.predict(data)
        score = gauss.score(data)
        gauss_scores.append(score)
        sil = silhouette_score(X=data, labels=labels)
        sil_scores.append(sil)
        cal = calinski_harabasz_score(X=data, labels=labels)
        cal_scores.append(cal)
        dav = davies_bouldin_score(X=data, labels=labels)
        dav_scores.append(dav)
    return gauss_scores, sil_scores, cal_scores, dav_scores

def clusters_spectral(data, max_clusters=10):
    sil_scores = []
    cal_scores = []
    dav_scores = []
    for i in range(2,max_clusters+1):
        sc = SpectralClustering(i, random_state=72).fit(data)
        labels = sc.labels_
        sil = silhouette_score(X=data, labels=labels)
        sil_scores.append(sil)
        cal = calinski_harabasz_score(X=data, labels=labels)
        cal_scores.append(cal)
        dav = davies_bouldin_score(X=data, labels=labels)
        dav_scores.append(dav)
    return sil_scores, cal_scores, dav_scores

def val_test_scores(model):
    global X_train, y_train_label, X_test, y_test_label
    scores = cross_val_score(estimator = model, X= X_train, y= y_train_label, scoring='accuracy', cv=5)
    model.fit(X_train, y_train_label)
    test_score = model.score(X_test, y_test_label)
    return scores, test_score

def modelo_xgboost_np(param_space):
    global X_train, y_train_label, X_test, y_test_label
    # Modelo XGBoosting sin preprocesar datos
    @scheduler.parallel(n_jobs=-1)
    def objective(**params):
        global X_train, y_train_label
        model = XGBClassifier(**params)
        score= cross_val_score(estimator = model, X= X_train, y= y_train_label, scoring='accuracy', cv=5).mean()
        return score
    tuner = Tuner(param_space, objective)
    best_results = tuner.maximize()
    print('best parameters np:', best_results['best_params'])
    print('best accuracy np:', best_results['best_objective'])
    # Scores de validación y prueba
    params = best_results['best_params']
    model = XGBClassifier(**params)
    scores, test_score = val_test_scores(model=model)
    return params, scores, test_score

def modelo_xgboost_sc(param_space, preprocessor):
    global X_train, y_train_label, X_test, y_test_label
    @scheduler.parallel(n_jobs=-1)
    def objective(**params):
        global X_train, y_train_label, preprocessor
        model = Pipeline([('preprocessing', preprocessor),('xg', XGBClassifier(**params))])
        score = cross_val_score(estimator = model, X= X_train, y= y_train_label, scoring='accuracy', cv=5).mean()
        return score
    tuner = Tuner(param_space, objective)
    best_results = tuner.maximize()
    print('best parameters:', best_results['best_params'])
    print('best accuracy:', best_results['best_objective'])
    params = best_results['best_params']
    model = Pipeline([('preprocessing', preprocessor),('xg', XGBClassifier(**params))])
    scores, test_score = val_test_scores(model=model)
    return params, scores, test_score

# Extracción sujetos mal clasificados
def errores(model, label):
    global X_train, y_train, X_test, y_test
    skf = StratifiedKFold(n_splits=5)

    le = LabelEncoder()
    le.fit(y_train)
    y_train_label = le.fit_transform(y_train)
    y_test_label = le.fit_transform(y_test)

    df_errados = pd.DataFrame(columns=['predicted'])
    # errores dataset de entrenamiento
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train_label)):
        model.fit(X_train.iloc[train_index], y_train_label[train_index])
        y_est = model.predict(X_train.iloc[test_index])
        errado = test_index[y_train_label[test_index] != y_est]
        y_pred = le.inverse_transform(y_est)
        y_p_df = pd.DataFrame(data=(y_pred), index=test_index, columns=['predicted'])
        errado_idx = pd.Index(errado)
        y_errados = y_p_df.loc[errado_idx].copy()
        df_errados = pd.concat([df_errados, y_errados], ignore_index=False)
    y_t = y_train.reset_index().copy()
    df_errados = pd.merge(y_t, df_errados, how='inner', left_index=True, right_index=True)
    df_errados.set_index('subject', inplace=True)

    # errores dataset de prueba
    y_test_pred = model.predict(X_test)
    y_test_pred = le.inverse_transform(y_test_pred)
    y_test_pred_df = pd.DataFrame(data=y_test_pred, index=y_test.index, columns=['predicted'])
    test_errados_df = pd.merge(y_test, y_test_pred_df, left_index=True, right_index=True)
    test_errados_df = test_errados_df[test_errados_df[label] != test_errados_df['predicted']]

    df_errados = pd.concat([df_errados, test_errados_df], ignore_index=False)
    return df_errados

# Entrenamiento de mejor modelo
def mejor_modelo(params, X, y, pre_pipe):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77, stratify=y)
    y_train_label = LabelEncoder().fit_transform(y_train)
    y_test_label = LabelEncoder().fit_transform(y_test)

    if pre_pipe == 'np':
        pipe = XGBClassifier(**params)
        scores = cross_val_score(estimator = pipe, X= X_train, y= y_train_label, scoring='accuracy', cv=5)
        print('mean val score: ', scores.mean())
        print('std val score: ', scores.std())
        model_fit = pipe.fit(X_train, y_train_label)
        print('test score: ',model_fit.score(X_test, y_test_label))
    elif pre_pipe == 'sc':
        # separación de variables para preprocesar
        continuas_cols = X_train.select_dtypes(include=['float64']).columns.to_list()
        discretas_cols = X_train.select_dtypes(include=['int64']).columns.to_list()
        preprocessor = ColumnTransformer([('sc', StandardScaler(), continuas_cols), ('min_max', MinMaxScaler(), discretas_cols)], remainder='passthrough')
        pipe = Pipeline([('preprocessing', preprocessor),('model', XGBClassifier(**params))])
        scores = cross_val_score(estimator = pipe, X= X_train, y= y_train_label, scoring='accuracy', cv=5)
        print('mean val score: ', scores.mean())
        print('std val score: ', scores.std())
        model_fit = pipe.fit(X_train, y_train_label)
        print('test score: ',model_fit.score(X_test, y_test_label))
    elif pre_pipe == 'pt':
        # separación de variables para preprocesar
        continuas_cols = X_train.select_dtypes(include=['float64']).columns.to_list()
        discretas_cols = X_train.select_dtypes(include=['int64']).columns.to_list()
        preprocessor = ColumnTransformer([('pt', PowerTransformer(), continuas_cols), ('min_max', MinMaxScaler(), discretas_cols)], remainder='passthrough')
        pipe = Pipeline([('preprocessing', preprocessor),('model', XGBClassifier(**params))])
        scores = cross_val_score(estimator = pipe, X= X_train, y= y_train_label, scoring='accuracy', cv=5)
        print('mean val score: ', scores.mean())
        print('std val score: ', scores.std())
        model_fit = pipe.fit(X_train, y_train_label)
        print('test score: ',model_fit.score(X_test, y_test_label))
    else:
        print('error de selección de pipeline de preprocesamiento')
    return pipe, model_fit

# Extracción sujetos mal clasificados
def errores(model, label, X_train, y_train, X_test, y_test):
    skf = StratifiedKFold(n_splits=5)
    le = LabelEncoder()
    le.fit(y_train)
    y_train_label = le.fit_transform(y_train)
    y_test_label = le.fit_transform(y_test)
    df_errados = pd.DataFrame(columns=['predicted'])
    # errores dataset de entrenamiento
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train_label)):
        model.fit(X_train.iloc[train_index], y_train_label[train_index])
        y_est = model.predict(X_train.iloc[test_index])
        errado = test_index[y_train_label[test_index] != y_est]
        y_pred = le.inverse_transform(y_est)
        y_p_df = pd.DataFrame(data=(y_pred), index=test_index, columns=['predicted'])
        errado_idx = pd.Index(errado)
        y_errados = y_p_df.loc[errado_idx].copy()
        df_errados = pd.concat([df_errados, y_errados], ignore_index=False)
    y_t = y_train.reset_index().copy()
    df_errados = pd.merge(y_t, df_errados, how='inner', left_index=True, right_index=True)
    df_errados.set_index('subject', inplace=True)
    # errores dataset de prueba
    y_test_pred = model.predict(X_test)
    y_test_pred = le.inverse_transform(y_test_pred)
    y_test_pred_df = pd.DataFrame(data=y_test_pred, index=y_test.index, columns=['predicted'])
    test_errados_df = pd.merge(y_test, y_test_pred_df, left_index=True, right_index=True)
    test_errados_df = test_errados_df[test_errados_df[label] != test_errados_df['predicted']]
    # concatenación de errores
    df_errados = pd.concat([df_errados, test_errados_df], ignore_index=False)
    return df_errados
