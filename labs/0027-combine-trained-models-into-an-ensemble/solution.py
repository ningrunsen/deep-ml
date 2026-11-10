# --- cell 1 ---
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_ensemble(base_models, X_val, y_val):

    def _predict_hv(X):
        predictions = np.array([
            m.predict(X) for m in base_models
        ])

        return np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=0,
            arr=predictions
        )

    def _get_proba(X):
        return np.array([
            m.predict_proba(X) for m in base_models
        ])

    def _predict_sv(X):
        predictions = _get_proba(X)
        return np.argmax(np.mean(predictions, axis=0), axis=1)

    # Split validation data into fitting and selection sets
    X_fit, X_select, y_fit, y_select = train_test_split(
        X_val, y_val,
        test_size=0.3,
        random_state=42,
        stratify=y_val
    )

    # Learn weighted-voting weights
    accuracies = np.array([
        accuracy_score(y_fit, m.predict(X_fit))
        for m in base_models
    ])

    if accuracies.sum() == 0:
        weights = np.ones(len(base_models)) / len(base_models)
    else:
        weights = accuracies / accuracies.sum()

    def _predict_wv(X):
        predictions = _get_proba(X)
        weighted = predictions * weights[:, None, None]

        return np.argmax(np.sum(weighted, axis=0), axis=1)

    # Train stacking meta-model
    X_meta = np.concatenate([
        m.predict_proba(X_fit)
        for m in base_models
    ], axis=1)

    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_meta, y_fit)

    def _predict_stacking(X):
        X_meta = np.concatenate([
            m.predict_proba(X)
            for m in base_models
        ], axis=1)

        return meta_model.predict(X_meta)

    # Evaluate on data not used to fit the combiner
    voters = {
        "hv": _predict_hv,
        "sv": _predict_sv,
        "wv": _predict_wv,
        "stacking": _predict_stacking
    }

    accuracies = {
        name: accuracy_score(y_select, fn(X_select))
        for name, fn in voters.items()
    }

    best_voter = max(accuracies, key=accuracies.get)

    def predict(X):
        return voters[best_voter](X)

    return predict