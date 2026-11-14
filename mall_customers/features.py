from sklearn.preprocessing import StandardScaler
from mall_customers.config import FEATURES


def prepare_features(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    return X_scaled, scaler