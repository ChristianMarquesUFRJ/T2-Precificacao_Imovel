#################################################################################
# Universidade Federal do Rio de Janeiro
# Disciplina: Introdução ao Aprendizado de Máquina - EEL891
# Professor: Heraldo L. S. Almeida
# Desenvolvedor: Chritian Marques de Oliveira Silva
# DRE: 117.214.742
# Trabalho 1: Classificação - Sistema de apoio à decisão p/ aprovação de crédito
#################################################################################

#################################################################################
# Importação de bibliotecas
#################################################################################
from os import X_OK
import pandas   as pd
import numpy    as np
import seaborn  as sns; sns.set()
from sklearn.neighbors          import KNeighborsRegressor
from sklearn.preprocessing      import MinMaxScaler, PolynomialFeatures
from matplotlib                 import pyplot as plt
from sklearn.model_selection    import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble           import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics            import accuracy_score, precision_score, r2_score
from sklearn.metrics            import recall_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.metrics            import plot_precision_recall_curve, plot_roc_curve
from sklearn.linear_model       import LinearRegression, Ridge, Lasso, SGDRegressor, LogisticRegression
from sklearn.svm                import SVR
from sklearn.feature_selection  import mutual_info_regression

#################################################################################
# Decisão se o código vai rodar como predição ou validação
#################################################################################
# MODE_VALIDATION = True
# MODE_CROSS_VALIDATION = False

# MODE_VALIDATION = False
# MODE_CROSS_VALIDATION = True

MODE_VALIDATION = False
MODE_CROSS_VALIDATION = False

#################################################################################
# Leitura dos arquivos de input
#################################################################################
def get_data(path):
    return pd.read_csv(path)

#################################################################################
# Preprocessamento
#################################################################################

#-------------------------------------------------------------------------------
# Visualização da correlação dos parâmetros usados
#-------------------------------------------------------------------------------
def show_correlation_matrix(data):
    print("\n\n")
    print("MATRIZ DE CORRELAÇÃO:")
    corrMatrix = data.corr()
    sns.heatmap(corrMatrix, xticklabels=1, yticklabels=1, vmin=0, vmax=1)
    plt.show()
    print("\n\n")

#-------------------------------------------------------------------------------
# Elimincação das colunas não utilizadas
#-------------------------------------------------------------------------------
def filter_best_params(data, is_train):
    selected_params = [
        'tipo',
        'bairro',
        'tipo_vendedor',
        'quartos',
        'suites',
        'vagas',
        'area_util',
        'area_extra',
        'diferenciais'
        # 'churrasqueira',
        # 'estacionamento',
        # 'piscina',
        # 'playground',
        # 'quadra',
        # 's_festas',
        # 's_jogos',
        # 's_ginastica',
        # 'sauna',
        # 'vista_mar'
        ]

    output = ['preco']

    if (is_train):
        selected_params.extend(output)
        return data[selected_params]
    
    return data[selected_params]

#-------------------------------------------------------------------------------
# Organização dos dados
#-------------------------------------------------------------------------------

# ALTERAÇÃO DE DADOS
# --------------------------------------
def pretrain_change_data(data):

    # Colunas auxiliares com os diferenciais divididos
    data['diferenciais_1'] = ["nenhum"] * len(data)
    data['diferenciais_2'] = ["nenhum"] * len(data)

    # Split dos diferenciais nas duas colunas auxiliares criadas
    for i in range (len(data)):
        split = data['diferenciais'][i].split(" e ", 2)
        if (len(split) == 2):
            data['diferenciais_1'][i], data['diferenciais_2'][i] = split
        else:
            data['diferenciais_1'][i] = split[0]

    # ONE-HOT ENCODING manual (juntando informações das 2 colunas de dados criadas)
    data['d_piscina'] = np.select([(data['diferenciais_1'] == 'piscina'), (data['diferenciais_2'] == 'piscina')], [1, 1])
    data['d_copa'] = np.select([(data['diferenciais_1'] == 'copa'), (data['diferenciais_2'] == 'copa')], [1, 1])
    data['d_churrasqueira'] = np.select([(data['diferenciais_1'] == 'churrasqueira'), (data['diferenciais_2'] == 'churrasqueira')], [1, 1])
    data['d_playground'] = np.select([(data['diferenciais_1'] == 'playground'), (data['diferenciais_2'] == 'playground')], [1, 1])
    data['d_sauna'] = np.select([(data['diferenciais_1'] == 'sauna'), (data['diferenciais_2'] == 'sauna')], [1, 1])
    data['d_quad_poli'] = np.select([(data['diferenciais_1'] == 'quadra poliesportiva'), (data['diferenciais_2'] == 'quadra poliesportiva')], [1, 1])
    data['d_sal_fest'] = np.select([(data['diferenciais_1'] == 'salao de festas'), (data['diferenciais_2'] == 'salao de festas')], [1, 1])
    data['d_camp_fut'] = np.select([(data['diferenciais_1'] == 'campo de futebol'), (data['diferenciais_2'] == 'campo de futebol')], [1, 1])
    data['d_est_visit'] = np.select([(data['diferenciais_1'] == 'estacionamento visitantes'), (data['diferenciais_2'] == 'estacionamento visitantes')], [1, 1])
    data['d_esquina'] = np.select([(data['diferenciais_1'] == 'esquina'), (data['diferenciais_2'] == 'esquina')], [1, 1])
    data['d_sal_gin'] = np.select([(data['diferenciais_1'] == 'sala de ginastica'), (data['diferenciais_2'] == 'sala de ginastica')], [1, 1])
    data['d_frente_mar'] = np.select([(data['diferenciais_1'] == 'frente para o mar'), (data['diferenciais_2'] == 'frente para o mar')], [1, 1])
    data['d_sal_jogos'] = np.select([(data['diferenciais_1'] == 'salao de jogos'), (data['diferenciais_2'] == 'salao de jogos')], [1, 1])
    data['d_children_care'] = np.select([(data['diferenciais_1'] == 'children care'), (data['diferenciais_2'] == 'children care')], [1, 1])
    data['d_quad_squash'] = np.select([(data['diferenciais_1'] == 'quadra de squash'), (data['diferenciais_2'] == 'quadra de squash')], [1, 1])
    data['d_vestiario'] = np.select([(data['diferenciais_1'] == 'vestiario'), (data['diferenciais_2'] == 'vestiario')], [1, 1])
    data['d_hidromassagem'] = np.select([(data['diferenciais_1'] == 'hidromassagem'), (data['diferenciais_2'] == 'hidromassagem')], [1, 1])

    # Remoção das colunas auxiliares
    data = data.drop(['diferenciais', 'diferenciais_1','diferenciais_2'], axis=1)

    return data
    
# ONE-HOT ENCODING
# --------------------------------------
def pretrain_data_one_hot_encoding(data):
    one_hot_encoding_params = [
        'tipo',
        'tipo_vendedor',
        'bairro'
        ]
    data = pd.get_dummies(data,columns=one_hot_encoding_params)
    return data

def pretrain_categorical_data_formater(data):
    one_hot_encoding_params = [
        'tipo',
        'tipo_vendedor',
        'bairro'
        ]
    for label, content in data.items():
        if (label in one_hot_encoding_params):
            data[label] = pd.Categorical(content).codes+1
    return data

# Move a coluna target para a ultima coluna
# --------------------------------------
def move_price_to_end(data, target_col):
    # print("Tamanho: ", len(data.T))
    # print(data)
    price_col = data.pop(target_col)
    data.insert(len(data.T), target_col, price_col)
    # print(data)
    return data

#################################################################################
# Preparação dos dados para o treinamento
#################################################################################

#-------------------------------------------------------------------------------
# Embaralhamento dos dados
#-------------------------------------------------------------------------------
def shuffle_data(data):
    return data.sample(frac=1,random_state=0)

#-------------------------------------------------------------------------------
# Adição de dados a mais que existem no treino e não existem no teste, e vice versa
#-------------------------------------------------------------------------------
def add_difference_param_train_test(my_data, other_data):
    params = my_data.columns.difference(other_data.columns)
    params = params.to_numpy().tolist()

    if ('preco' in params):
        params.remove('preco')

    for param in params:
        other_data[param] = [0] * (len(other_data.index))

    return other_data

# ------------------------------------------------------------------------------
# Divisão entre inputs e outputs
# ------------------------------------------------------------------------------
def split_inputs_outputs(data):
    x = data.loc[:,data.columns!='preco']
    y = data.loc[:,data.columns=='preco']
    return x, y

def concat_train_test(train, test):
    return pd.concat([train, test], ignore_index=True)

def split_train_test(data, row):
    train = data.loc[:row-1,:]
    test = data.loc[row:,:]
    return train, test

# ------------------------------------------------------------------------------
# Ajustar a escala dos atributos nos conjuntos de treino e de teste
# ------------------------------------------------------------------------------
# def adjust_scale(data, model):
#     return model.transform(data)
def adjust_scale(data):
    scale_adjust = MinMaxScaler()
    scale_adjust.fit(data)
    data[data.columns] = scale_adjust.transform(data[data.columns])
    return data



#################################################################################
# Processamento: Treinamento e Predição
#################################################################################

#-------------------------------------------------------------------------------
# Treinamento do classificador com o conjunto de treino
#-------------------------------------------------------------------------------
def train_KNN(x_train, y_train, _n_neighbors, _p):
    model = KNeighborsRegressor(
        n_neighbors = _n_neighbors,
        weights     = 'uniform',
        p           = _p)
    return model.fit(x_train,y_train)

def train_Random_Forest(x_train, y_train, depth):
    model = RandomForestRegressor(
            max_depth=depth, 
            random_state=0)
    return model.fit(x_train,y_train)

def adjust_params_Polynomial_Regression(x, deg):
    pf = PolynomialFeatures(degree=deg)
    return pf.fit_transform(x)

def train_Linear_Regression(x_train, y_train):
    model = LinearRegression()
    return model.fit(x_train, y_train)

def train_Logistic_Regression(x_train, y_train, c, tolerance):
    model = LogisticRegression(
        random_state=0,
        C=c,
        solver='lbfgs', # newton-cg, lbfgs, liblinear, sag, saga
        penalty='l2', # l2, l1, elasticnet
        tol=tolerance,
        max_iter=100000,
    )
    return model.fit(x_train, y_train)

def train_Lasso(x_train, y_train, alp):
    model = Lasso(
            alpha=alp, 
            random_state=0)
    return model.fit(x_train,y_train)

def train_Ridge(x_train, y_train, alp):
    model = Ridge(
            alpha=alp, 
            random_state=0)
    return model.fit(x_train,y_train)

def train_SGD(x_train, y_train, alp, tolerance):
    model = SGDRegressor(
            alpha=alp, 
            loss='squared_loss', # squared_loss, huber, epsilon_insensitive, squared_epsilon_insensitive
            penalty='elasticnet', # l2, l1, elasticnet
            tol=tolerance,
            max_iter=100000,
            random_state=0)
    return model.fit(x_train,y_train)

def train_SVR_linear(x_train, y_train):
    model = SVR(
            kernel='linear', 
            C=1000.0)
    return model.fit(x_train,y_train)

def train_SVR_poly(x_train, y_train):
    model = SVR(
            kernel='poly',
            degree=2,
            C=1000.0)
    return model.fit(x_train,y_train)

def train_SVR_RBF(x_train, y_train):
    model = SVR(
            kernel='rbf',
            gamma=0.85,
            C=1000.0)
    return model.fit(x_train,y_train)

def train_GridSearchCV(x_train, y_train):
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = SVR()
    model = GridSearchCV(svr, parameters)
    return model.fit(x_train,y_train)

def train_GradientBoostingRegressor(x_train, y_train):
    model = GradientBoostingRegressor(random_state=0)
    return model.fit(x_train,y_train)

def train_AdaBoostRegressor(x_train, y_train):
    model = AdaBoostRegressor(
        random_state=0,
        n_estimators=65,
        learning_rate=0.75,
        loss='exponential' # linear, square, exponential
        )
    return model.fit(x_train,y_train)

#-------------------------------------------------------------------------------
# Predição do resultado com o classificador treinado
#-------------------------------------------------------------------------------
def predict(model, data):
    return model.predict(data)

#-------------------------------------------------------------------------------
# Validação do sistema com os dados usados (fazendo uso do treinamento cruzado)
#-------------------------------------------------------------------------------
def validation(x, y):
    print('\n\n\n')
    print ( "\nVALIDAÇÃO DO MODELO")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size = 0.2,
        random_state = 0   
        )

    # --------
    # Treinamento com KNN
    k, p = 5, 1
    model_trained = train_KNN(x_train, y_train, k, p)

    # Treinamento com Random Forest
    # depth = 25
    # model_trained = train_Random_Forest(x_train, y_train, depth)

    # Treinamento com Regressão com regularização Lasso
    # degree = 2
    # x_train = adjust_params_Polynomial_Regression(x_train, degree)
    # x_test = adjust_params_Polynomial_Regression(x_test, degree)
    # Regressão linear
    # model_trained = train_Linear_Regression(x_train, y_train)
    # Lasso
    # alpha = 0.001
    # model_trained = train_Lasso(x_train, y_train, alpha)
    # Ridge
    # alpha = 0.001
    # model_trained = train_Ridge(x_train, y_train, alpha)
    # SGD
    # alpha, tolerance = 0.001, 1e-6
    # model_trained = train_SGD(x_train, y_train, alpha, tolerance)
    # Regressão Logística
    # C, tolerance = 1.0, 1e-5
    # model_trained = train_Logistic_Regression(x_train, y_train, C, tolerance)
    
    # --------

    # Predição
    y_predict_test = predict(model_trained, x_test)

    # Indicação da acurácia do treino
    # scoring(y_train, y_predict_train)
    # scoring(y_test, y_predict_test)

    show_metrics(y_test, y_predict_test)
    rmse, r2 = get_error_metrics (y_test, y_predict_test)
    print('\n K =%2d  RMSE = %2.4f  R2 = %2.4f' % (k, rmse, r2))
        
def cross_validation_KNN(x, y):
    print('\n\n\n')
    print ( "\nVALIDAÇÃO CRUZADA DO MODELO")
    print ( "\n  K   ACERTO(%)")
    print ( " --   ------")
    cross_val = 5
    for k in range(501,550,2):
        regressor = KNeighborsRegressor(
            n_neighbors = k,
            weights     = 'uniform',
            p           = 1)

        scores = cross_val_score(regressor, x, y, cv=cross_val)
        
        print ('k = %2d' % k, 'Acurácia média = %6.1f' % (100*sum(scores)/cross_val))
        
def cross_validation_Random_Forest(x, y):
    print('\n\n\n')
    print ( "\nVALIDAÇÃO CRUZADA DO MODELO")
    print ( "\n  D   ACERTO(%)")
    print ( " --   ------")

    cross_val = 5

    for depth in range(15,30):
        regressor = train_Random_Forest(x, y, depth)

        scores = cross_val_score(regressor, x, y, cv=cross_val)
        
        print (' %2d' % depth, 'Acurácia média = %6.1f' % (100*sum(scores)/cross_val))
        
def cross_validation_Polynomial_Regression(x, y):
    print('\n\n\n')
    print ( "\nVALIDAÇÃO CRUZADA DO MODELO")
    print ( "\n  G   ACERTO(%)")
    print ( " --   ------")

    for degree in range(1,3):
        x = adjust_params_Polynomial_Regression(x, degree)
        # Regressão linear
        model = train_Linear_Regression(x, y)
        # Lasso
        # alpha = 0.001
        # model = train_Lasso(x, y, alpha)
        # Ridge
        # alpha = 0.001
        # model = train_Ridge(x, y, alpha)
        # SGD
        # alpha, tolerance = 0.001, 1e-6
        # model = train_SGD(x, y, alpha, tolerance)
        # Regressão Logística
        # C, tolerance = 1.0, 1e-5
        # model_trained = train_Logistic_Regression(x, y, C, tolerance)

        y_pred = predict(model, x)
        rmse, r2 = get_error_metrics (y, y_pred)
        print('\n Degree =%2d  RMSE = %2.4f  R2 = %2.4f' % (degree, rmse, r2))

def cross_validation_SVR_Regression(x, y):
    print('\n\n\n')
    print ( "\nVALIDAÇÃO CRUZADA DO MODELO SVR")

    model = train_SVR_linear(x, y)
    y_pred = predict(model, x)
    rmse, r2 = get_error_metrics (y, y_pred)
    print('\n LINEAR  RMSE = %2.4f  R2 = %2.4f' % (rmse, r2))

    model = train_SVR_poly(x, y)
    y_pred = predict(model, x)
    rmse, r2 = get_error_metrics (y, y_pred)
    print('\n POLY  RMSE = %2.4f  R2 = %2.4f' % (rmse, r2))

    model = train_SVR_RBF(x, y)
    y_pred = predict(model, x)
    rmse, r2 = get_error_metrics (y, y_pred)
    print('\n RBF  RMSE = %2.4f  R2 = %2.4f' % (rmse, r2))



#################################################################################
# Pós processamento
#################################################################################
#-------------------------------------------------------------------------------
# Exportar os resultados preditos em um arquivo ".csv" no formato correto
#-------------------------------------------------------------------------------
def export_to_csv(y_predict, start_id_count):
    # Cria a coluna com os indices corretos de saida
    y_id = np.array(np.arange(start_id_count, start_id_count+len(y_predict), 1).tolist())

    y_formated = ["%.2f"%item for item in y_predict]

    # Converte os arrays em dataframe
    DF = pd.DataFrame(data={
        'Id': y_id,
        'preco': y_formated
        })
    
    # Salva o dataframe em um arquivo .csv sem a primeira coluna ser o indice padrão [0 a len(y_predict)]
    DF.to_csv("data.csv", index=False)

def show_metrics(y_true, y_pred):
    print (" Acurácia           : %6.1f %%" % (100*accuracy_score(y_pred,y_true)) )
    print ("  " )   
    print (" Precisão           : %6.1f %%" % (100*precision_score(y_pred,y_true)) )
    print ("  " )   
    print (" Sensibilidade      : %6.1f %%" % (100*recall_score(y_pred,y_true)) )
    print ("  " )   
    print (" Score F1           : %6.1f %%" % (100*f1_score(y_pred,y_true)) )
    print ("  " )   
    print (" Área sob ROC       : %6.1f %%" % (100*roc_auc_score(y_pred,y_true)) )


def get_error_metrics (y_true, y_pred):
    mse  = mean_squared_error(y_pred, y_true)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_pred, y_true)
    return rmse, r2

def plot_curves(model, x_test, y_pred):
    plot_roc_curve(model, x_test, y_pred)
    plot_precision_recall_curve(model, x_test, y_pred)

#################################################################################
# Funções gerais de lógica
#################################################################################
def preprocessing(data):
    data = pretrain_change_data(data)
    # data = pretrain_data_one_hot_encoding(data)
    # data = shuffle_data(data)
    return data

if __name__ == "__main__":
    pd.set_option("mode.chained_assignment", None)

    # Le os dados dos arquivos e transforma em dataframes
    input_train_data = get_data('data\conjunto_de_treinamento.csv')
    input_test_data = get_data('data\conjunto_de_teste.csv')

    # Escolhe os mehlores parâmetros
    train_data = filter_best_params(input_train_data, True)
    test_data = filter_best_params(input_test_data, False)

    # Realiza toda a organização, formatação e configuração dos dados
    train_data = preprocessing(train_data)
    test_data = preprocessing(test_data)
    
    # Remove atributos que ou o treino nao tem, ou o teste
    test_data = add_difference_param_train_test(train_data, test_data)
    train_data = add_difference_param_train_test(test_data, train_data)

    # Mostra a relação entre os parâmetros
    show_correlation_matrix(train_data)

    # Alinha todos os parametros em ordem alfabetica
    train_data = train_data.reindex(sorted(train_data.columns), axis=1)
    test_data = test_data.reindex(sorted(test_data.columns), axis=1)

    # Move a coluna de preco para a ultima coluna
    train_data = move_price_to_end(train_data, 'preco')

    # Split dos dados de input e outout do treinamento
    x_train, y_train = split_inputs_outputs(train_data)
    x_test = test_data
    y_train = y_train.values.ravel()

    # ---
    x_train_rows = len(x_train)
    x_train_0, x_test_0 = x_train, x_test
    data = concat_train_test(x_train, x_test)
    data = pretrain_categorical_data_formater(data)
    data_0 = data.copy()
    data = adjust_scale(data)
    x_train, x_test = split_train_test(data, x_train_rows)

    # Obtenção dos parametros que influenciam no resultado final
    mutual_info = mutual_info_regression(x_train, y_train, random_state=42, n_neighbors=10)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = x_train.columns
    mutual_info = mutual_info.sort_values(ascending=False)
    mutual_info.plot.bar(figsize=(15,5))
    # Seleção dos parametros que influenciam no resultado final
    selected_top_columns = []
    for x in range(len(mutual_info)):
        if (mutual_info[x] > 0.0):
            selected_top_columns.append(mutual_info.index[x])
    x_train = x_train[selected_top_columns]
    x_test = x_test[selected_top_columns]

    if (MODE_VALIDATION):
        validation(x_train, y_train)
    elif (MODE_CROSS_VALIDATION):
        # cross_validation_KNN(x_train, y_train)
        cross_validation_Random_Forest(x_train, y_train)
        # cross_validation_Polynomial_Regression(x_train, y_train)
        # cross_validation_SVR_Regression(x_train, y_train)
    else:
        # model_trained = train_KNN(x_train, y_train, 10, 1)
        # model_trained = train_SVR_linear(x_train, y_train)
        # model_trained = train_SVR_poly(x_train, y_train)
        # model_trained = train_SVR_RBF(x_train, y_train)

        # model_trained = train_GridSearchCV(x_train, y_train)
        # model_trained = train_GradientBoostingRegressor(x_train, y_train)
        model_trained = train_AdaBoostRegressor(x_train, y_train)
        
        # depth = 15
        # model_trained = train_Random_Forest(x_train, y_train, depth)

        # Predição
        y_predict_train = predict(model_trained, x_train)
        y_predict_test = predict(model_trained, x_test)
        
        # Indicação das métricas
        rmse, r2 = get_error_metrics (y_train, y_predict_train)
        print('\n RMSE = %2.4f  R2 = %2.4f' % (rmse/1000000.0, r2))

        # Plot valores

        # Exportação do resultado final da predição para a planilha de resposta
        export_to_csv(y_predict_test, 0)