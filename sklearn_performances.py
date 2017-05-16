import numpy as np
from sklearn.model_selection import KFold


def get_confidence_interval(model, row, confidence=0.95):
    confidence_mapping = {
        0.80: 1.25,
        0.90: 1.645,
        0.95: 1.96,
        0.98: 2.33,
        0.99: 2.58
    }

    z_value = confidence_mapping[confidence]

    prediction_vector = []
    for tree in model.estimators_:
        prediction_vector.append(tree.predict(row)[0])

    mean_prediction = np.mean(prediction_vector)
    std_prediction = np.std(prediction_vector)

    conf_interval = (mean_prediction - z_value * (std_prediction / np.sqrt(len(model.estimators_))),
                     mean_prediction + z_value * (std_prediction / np.sqrt(len(model.estimators_))))

    return {
        "confidence_high": conf_interval[1],
        "confidence_low": conf_interval[0],
        "mean": mean_prediction,
        "std": std_prediction,
        "predictions": prediction_vector
    }


def get_perfs(model, X, y, classification=False):
    predictions = []
    trues = []
    predict_proba = []
    kf = KFold(n_splits=6)
    for train_index, test_index in tqdm(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        predictions.extend(model.predict(X_test))
        trues.extend(y_test)
        if classification:
            predict_proba.extend(model.predict_proba(X_test))

    dict_ = {
        "predictions": predictions,
        "trues": trues
    }
    if classification:
        dict_["predprobas"] = predict_proba

    return dict_
