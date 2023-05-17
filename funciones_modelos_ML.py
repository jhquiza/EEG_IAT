import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.inspection import permutation_importance
from mango import Tuner, scheduler
from mango.domain.distribution import loguniform
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel, f_classif, mutual_info_classif
from xgboost import XGBClassifier

# Ajuste modelos de clustering KMeans
def clusters_kmeans(data, max_clusters=10):
    inertias = []
    sil_scores = []
    cal_scores = []
    dav_scores = []
    for i in range(2,max_clusters+1):
        kmeans = KMeans(i, random_state=1).fit(data)
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

# Ajuste modelos de clustering Gaussian Mixture
def clusters_gaussian(data, max_clusters=10):
    gauss_scores = []
    sil_scores = []
    cal_scores = []
    dav_scores = []
    for i in range(2,max_clusters+1):
        gauss = GaussianMixture(i, random_state=1).fit(data)
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

# Ajuste modelos de clustering Spectral
def clusters_spectral(data, max_clusters=10):
    sil_scores = []
    cal_scores = []
    dav_scores = []
    for i in range(2,max_clusters+1):
        sc = SpectralClustering(i, random_state=1).fit(data)
        labels = sc.labels_
        sil = silhouette_score(X=data, labels=labels)
        sil_scores.append(sil)
        cal = calinski_harabasz_score(X=data, labels=labels)
        cal_scores.append(cal)
        dav = davies_bouldin_score(X=data, labels=labels)
        dav_scores.append(dav)
    return sil_scores, cal_scores, dav_scores

# Cálculo de scores de validación y prueba
def val_test_scores(model, X_train, y_train_label, X_test, y_test_label):
    scores = cross_val_score(estimator = model, X= X_train, y= y_train_label, scoring='f1_weighted', cv=5)
    model.fit(X_train, y_train_label)
    test_score = model.score(X_test, y_test_label)
    return scores, test_score

# Ajuste modelos XGBoosting sin preprocesar
def modelo_xgboost_np(X_train, y_train_label, X_test, y_test_label, param_space):
    @scheduler.parallel(n_jobs=-1)
    def objective(**params):
        model = XGBClassifier(**params)
        score= cross_val_score(estimator = model, X= X_train, y= y_train_label, scoring='f1_weighted', cv=5).mean()
        return score
    tuner = Tuner(param_space, objective)
    best_results = tuner.maximize()
    print('best parameters np:', best_results['best_params'])
    print('best f1_weighted np:', best_results['best_objective'])
    # Scores de validación y prueba
    params = best_results['best_params']
    model = XGBClassifier(**params)
    scores, test_score = val_test_scores(model=model, X_train=X_train, y_train_label=y_train_label, X_test=X_test, 
                                         y_test_label=y_test_label)
    return params, scores, test_score

# Ajuste modelos XGBoosting con preprocesado
def modelo_xgboost_sc(X_train, y_train_label, X_test, y_test_label, param_space, preprocessor):
    @scheduler.parallel(n_jobs=-1)
    def objective(**params):
        model = Pipeline([('preprocessing', preprocessor),('xg', XGBClassifier(**params))])
        score = cross_val_score(estimator = model, X= X_train, y= y_train_label, scoring='f1_weighted', cv=5).mean()
        return score
    tuner = Tuner(param_space, objective)
    best_results = tuner.maximize()
    print('best parameters:', best_results['best_params'])
    print('best f1_weighted:', best_results['best_objective'])
    params = best_results['best_params']
    model = Pipeline([('preprocessing', preprocessor),('xg', XGBClassifier(**params))])
    scores, test_score = val_test_scores(model=model, X_train=X_train, y_train_label=y_train_label, X_test=X_test, 
                                         y_test_label=y_test_label)
    return params, scores, test_score

# Entrenamiento de mejor modelo
def mejor_modelo(params, X, y, pre_pipe):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    y_train_label = LabelEncoder().fit_transform(y_train)
    y_test_label = LabelEncoder().fit_transform(y_test)

    if pre_pipe == 'np':
        pipe = XGBClassifier(**params)
        scores = cross_val_score(estimator = pipe, X= X_train, y= y_train_label, scoring='f1_weighted', cv=5)
        print('mean val score: ', scores.mean())
        print('std val score: ', scores.std())
        model_fit = pipe.fit(X_train, y_train_label)
        print('test score: ',model_fit.score(X_test, y_test_label))
    elif pre_pipe == 'sc':
        # separación de variables para preprocesar
        continuas_cols = X_train.select_dtypes(include=['float64']).columns.to_list()
        discretas_cols = X_train.select_dtypes(include=['int64']).columns.to_list()
        preprocessor = ColumnTransformer([('sc', StandardScaler(), continuas_cols), 
                                          ('min_max', MinMaxScaler(), discretas_cols)], remainder='passthrough')
        pipe = Pipeline([('preprocessing', preprocessor),('model', XGBClassifier(**params))])
        scores = cross_val_score(estimator = pipe, X= X_train, y= y_train_label, scoring='f1_weighted', cv=5)
        print('mean val score: ', scores.mean())
        print('std val score: ', scores.std())
        model_fit = pipe.fit(X_train, y_train_label)
        print('test score: ',model_fit.score(X_test, y_test_label))
    elif pre_pipe == 'pt':
        # separación de variables para preprocesar
        continuas_cols = X_train.select_dtypes(include=['float64']).columns.to_list()
        discretas_cols = X_train.select_dtypes(include=['int64']).columns.to_list()
        preprocessor = ColumnTransformer([('pt', PowerTransformer(), continuas_cols), 
                                          ('min_max', MinMaxScaler(), discretas_cols)], remainder='passthrough')
        pipe = Pipeline([('preprocessing', preprocessor),('model', XGBClassifier(**params))])
        scores = cross_val_score(estimator = pipe, X= X_train, y= y_train_label, scoring='f1_weighted', cv=5)
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

# Función para obtener el mejor modelo de clasificación sin preprocesar
def get_best_model_np(X_train, y_train_label, X_test, y_test_label):
    param_space_lr = dict(C=loguniform(-4,2), penalty=['l1','l2'])
    param_space_knn = dict(n_neighbors=np.arange(1, 20), weights=['uniform','distance'], p=[1,2])
    param_space_rf = dict(n_estimators=np.arange(1, 100), ccp_alpha=loguniform(-4,6))
    param_space_gb = dict(n_estimators=np.arange(1, 100), ccp_alpha=loguniform(-4,6), subsample=uniform(0,1))
    param_space_svc = dict(gamma=uniform(0, 1), C=loguniform(-4,8), kernel=['poly','rbf','sigmoid'], degree=range(1,5))
    param_space_xg =dict(n_estimators=range(1,100), max_depth=range(3,10), subsample=uniform(0.1,0.9), eta=uniform(0,1), 
                         colsample_bytree=uniform(0.1,0.9))

    def objective_lr(args_list):
        results = []
        for hyper_par in args_list:
            clf = LogisticRegression(solver='saga', max_iter=10000, random_state=1)
            clf.set_params(**hyper_par)
            result = cross_val_score(clf, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    def objective_knn(args_list):
        results = []
        for hyper_par in args_list:
            clf = KNeighborsClassifier()
            clf.set_params(**hyper_par)
            result = cross_val_score(clf, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    def objective_rf(args_list):
        results = []
        for hyper_par in args_list:
            clf = RandomForestClassifier(random_state=1)
            clf.set_params(**hyper_par)
            result = cross_val_score(clf, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    def objective_gb(args_list):
        results = []
        for hyper_par in args_list:
            clf = GradientBoostingClassifier(random_state=1)
            clf.set_params(**hyper_par)
            result = cross_val_score(clf, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    def objective_svc(args_list):
        results = []
        for hyper_par in args_list:
            clf = SVC(random_state=1)
            clf.set_params(**hyper_par)
            result = cross_val_score(clf, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    def objective_xg(args_list):
        results = []
        for hyper_par in args_list:
            clf = XGBClassifier()
            clf.set_params(**hyper_par)
            result = cross_val_score(clf, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    param_space_list = [param_space_lr, param_space_knn, param_space_rf, param_space_gb, param_space_svc, param_space_xg]
    objective_list = [objective_lr, objective_knn, objective_rf, objective_gb, objective_svc, objective_xg]
    model_names = ['Logistic Regression', 'KNN', 'Random Forest', 'Gradient Boosting', 'SVC', 'XGBoost']

    conf_dict = dict(num_iteration=40, domain_size=10000, initial_random=3)
    best_results = pd.DataFrame(index=['model', 'hiperparameters', 'mean cv score', 'std cv score', 'test score'])
    model_list = [LogisticRegression(solver='saga', max_iter=10000, random_state=1), KNeighborsClassifier(),
                   RandomForestClassifier(random_state=1), GradientBoostingClassifier(random_state=1), 
                   SVC(random_state=1), XGBClassifier()]
    for i in range(len(param_space_list)):
        param_space = param_space_list[i]
        objective = objective_list[i]
        model_name = model_names[i]
        tuner = Tuner(param_space, objective, conf_dict)
        print(tuner)
        best_model_results = tuner.maximize()
        params = best_model_results['best_params']
        model = model_list[i]
        model.set_params(**params)
        scores, test_score = val_test_scores(model=model, X_train=X_train, y_train_label=y_train_label, X_test=X_test, 
                                             y_test_label=y_test_label)
        temp = pd.DataFrame(data=[model_name, params, scores.mean(), scores.std(), test_score], 
                            index=['model', 'hiperparameters', 'mean cv score', 'std cv score', 'test score'])
        best_results = pd.concat([best_results, temp], axis=1)
    best_results = best_results.T
    best_results.set_index('model', inplace=True)
    return best_results

# Función para obtener el mejor modelo con preprocesamiento
def get_best_model_sc(preprocessor, model_names, X_train, y_train_label, X_test, y_test_label):
    param_space_lr = dict(C=loguniform(-4,2), penalty=['l1','l2'])
    param_space_knn = dict(n_neighbors=np.arange(1, 20), weights=['uniform','distance'], p=[1,2])
    param_space_rf = dict(n_estimators=np.arange(1, 100), ccp_alpha=loguniform(-4,6))
    param_space_gb = dict(n_estimators=np.arange(1, 100), ccp_alpha=loguniform(-4,6), subsample=uniform(0,1))
    param_space_svc = dict(gamma=uniform(0, 1), C=loguniform(-4,8), kernel=['poly','rbf','sigmoid'], degree=range(1,5))
    param_space_xg =dict(n_estimators=range(1,100), max_depth=range(3,10), subsample=uniform(0.1,0.9), 
                         eta=uniform(0,1), colsample_bytree=uniform(0.1,0.9))

    def objective_lr(args_list):
        results = []
        for hyper_par in args_list:
            clf = LogisticRegression(solver='saga', max_iter=10000, random_state=1)
            clf.set_params(**hyper_par)
            pipe = Pipeline([('preprocessing', preprocessor),('clf',clf)])
            result = cross_val_score(pipe, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    def objective_knn(args_list):
        results = []
        for hyper_par in args_list:
            clf = KNeighborsClassifier()
            clf.set_params(**hyper_par)
            pipe = Pipeline([('preprocessing', preprocessor),('clf',clf)])
            result = cross_val_score(pipe, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    def objective_rf(args_list):
        results = []
        for hyper_par in args_list:
            clf = RandomForestClassifier(random_state=1)
            clf.set_params(**hyper_par)
            pipe = Pipeline([('preprocessing', preprocessor),('clf',clf)])
            result = cross_val_score(pipe, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    def objective_gb(args_list):
        results = []
        for hyper_par in args_list:
            clf = GradientBoostingClassifier(random_state=1)
            clf.set_params(**hyper_par)
            pipe = Pipeline([('preprocessing', preprocessor),('clf',clf)])
            result = cross_val_score(pipe, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    def objective_svc(args_list):
        results = []
        for hyper_par in args_list:
            clf = SVC(random_state=1)
            clf.set_params(**hyper_par)
            pipe = Pipeline([('preprocessing', preprocessor),('clf',clf)])
            result = cross_val_score(pipe, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    def objective_xg(args_list):
        results = []
        for hyper_par in args_list:
            clf = XGBClassifier()
            clf.set_params(**hyper_par)
            pipe = Pipeline([('preprocessing', preprocessor),('clf',clf)])
            result = cross_val_score(pipe, X_train, y_train_label, scoring='f1_weighted', cv=5).mean()
            results.append(result)
        return results

    param_space_list = [param_space_lr, param_space_knn, param_space_rf, param_space_gb, param_space_svc, param_space_xg]
    objective_list = [objective_lr, objective_knn, objective_rf, objective_gb, objective_svc, objective_xg]
    
    conf_dict = dict(num_iteration=40, domain_size=10000, initial_random=3)
    best_results = pd.DataFrame(index=['model', 'hiperparameters', 'mean cv score', 'std cv score', 'test score'])
    model_list = [LogisticRegression(solver='saga', max_iter=10000, random_state=1), KNeighborsClassifier(), 
                  RandomForestClassifier(random_state=1), GradientBoostingClassifier(random_state=1), 
                  SVC(random_state=1), XGBClassifier()]
    for i in range(len(param_space_list)):
        param_space = param_space_list[i]
        objective = objective_list[i]
        model_name = model_names[i]
        tuner = Tuner(param_space, objective, conf_dict)
        best_model_results = tuner.maximize()
        params = best_model_results['best_params']
        clf= model_list[i]
        clf.set_params(**params)
        model = Pipeline([('preprocessing', preprocessor),('clf', clf)])
        scores, test_score = val_test_scores(model=model, X_train=X_train, y_train_label=y_train_label, X_test=X_test, 
                                             y_test_label=y_test_label)
        temp = pd.DataFrame(data=[model_name, params, scores.mean(), scores.std(), test_score], 
                            index=['model', 'hiperparameters', 'mean cv score', 'std cv score', 'test score'])
        best_results = pd.concat([best_results, temp], axis=1)
    best_results = best_results.T
    best_results.set_index('model', inplace=True)
    return best_results

# Funciones para selección de atributos
def select_features_clf(X_train, y_train, threshold='1.5*mean', mi_threshold=0.1):
    # Por selección por modelos
    # SVC
    X_train_df = X_train.copy()
    lsvc = LinearSVC(random_state=1).fit(X_train_df, y_train)
    model = SelectFromModel(lsvc, threshold=threshold, prefit=True)
    X_new = model.transform(X_train_df)
    features_lsvc = model.get_feature_names_out(input_features=X_train_df.columns)
    features_svc = pd.DataFrame(data=np.ones_like(features_lsvc), columns=['features_lsvc'], index=features_lsvc)

    # regresión logística l2
    lr = LogisticRegression(penalty="l2", solver='saga', max_iter=10000, random_state=1).fit(X_train_df, y_train)
    model = SelectFromModel(lr, threshold=threshold, prefit=True)
    X_new = model.transform(X_train_df)
    features_lrl2 = model.get_feature_names_out(input_features=X_train_df.columns)
    features_l2 = pd.DataFrame(data=np.ones_like(features_lrl2), columns=['features_lrl2'], index=features_lrl2)

    # regresión logística l1
    lr = LogisticRegression(penalty="l1", solver='saga', max_iter=10000, random_state=1).fit(X_train_df, y_train)
    model = SelectFromModel(lr, threshold=threshold, prefit=True)
    X_new = model.transform(X_train_df)
    features_lrl1 = model.get_feature_names_out(input_features=X_train_df.columns)
    features_l1 = pd.DataFrame(data=np.ones_like(features_lrl1), columns=['features_lrl1'], index=features_lrl1)

    # random forest
    rf = RandomForestClassifier(random_state=1).fit(X_train_df, y_train)
    model = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_new = model.transform(X_train_df)
    features_rf = model.get_feature_names_out(input_features=X_train_df.columns)
    features_rfo = pd.DataFrame(data=np.ones_like(features_rf), columns=['features_rf'], index=features_rf)

    # anova
    __, p_values = f_classif(X_train_df, y_train)
    features_anova = pd.DataFrame(p_values, columns=['p_values'], index=X_train_df.columns)
    features_anova = features_anova[features_anova['p_values']<0.05]
    features_anova['features_an'] = 1

    # información mutua
    mi = mutual_info_classif(X_train_df, y_train)
    features_mi = pd.DataFrame(mi, columns=['mutual information'], index=X_train_df.columns)
    features_mi = features_mi[features_mi['mutual information']>mi_threshold]
    features_mi['features_im'] = 1

    # atributos seleccionados
    features_sel = features_svc.join([features_l2, features_l1, features_rfo, features_anova, features_mi], how='outer')
    features_sel.drop(['p_values','mutual information'], axis=1, inplace=True)
    features_sel['total'] = features_sel.sum(axis=1)
    features_sel = features_sel[features_sel['total']>=3]
    lista_atributos = list(features_sel.index)
    return lista_atributos