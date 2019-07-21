from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Função para mudar a escala dos dados
def scale_feat(X_train, X_test, scaleType='min-max'):
    if scaleType=='min-max' or scaleType=='std':
        X_tr_norm = np.copy(X_train) # fazendo cópia para deixar original disponível
        X_ts_norm = np.copy(X_test)
        scaler = MinMaxScaler() if scaleType=='min-max' else StandardScaler()
        scaler.fit(X_tr_norm)
        X_tr_norm = scaler.transform(X_tr_norm)
        X_ts_norm = scaler.transform(X_ts_norm)
        return (X_tr_norm, X_ts_norm)
    else:
        raise ValueError("Tipo de escala não definida. Use 'min-max' ou 'std'.")
        
        
# convert dummies to multilabel
def dummie_to_multilabel(X):
    N = len(X)
    X_multi = np.zeros((N,1),dtype='int')
    for i in range(N):
        temp = np.where(X[i]==1)[0] # find where 1 is found in the array
        if temp.size == 0: # is a empty array, there is no '1' in the X[i] array
            X_multi[i] = 0 # so we denote this class '0'
        else:
            X_multi[i] = temp[0] + 1 # we have +1 because 
    return X_multi.T[0]