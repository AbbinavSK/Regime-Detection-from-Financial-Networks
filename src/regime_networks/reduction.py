from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from regime_networks import config


# Fit a StandardScaler and a PCA keeping PCA_VARIANCE of the variance on a train descriptor matrix; returns the reduced train matrix and the fitted transforms (the number of components kept is pca.n_components_).
def fit_reduction(X_train_raw):
    scaler = StandardScaler().fit(X_train_raw)
    scaled = scaler.transform(X_train_raw)
    pca = PCA(n_components=config.PCA_VARIANCE, random_state=config.SEED).fit(scaled)
    return pca.transform(scaled), scaler, pca


# Reduce a descriptor matrix with an already-fitted scaler and PCA.
def transform_reduction(X_raw, scaler, pca):
    return pca.transform(scaler.transform(X_raw))
