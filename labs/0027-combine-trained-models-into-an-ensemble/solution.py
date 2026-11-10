# --- cell 1 ---
import numpy as np
# You can import any sklearn module you need


def train_ensemble(base_models, X_val, y_val):
    """
    Combine K pre-trained base classifiers into an ensemble that predicts class
    labels on new data.

    The harness has already trained the base models on a separate training set.
    You DO NOT retrain them. Instead, use their predictions on (X_val, y_val) to
    figure out how to combine them, then return a `predict` callable that the
    harness will apply to held-out test data.

    Args:
        base_models: list of K trained sklearn-style classifiers. Each has:
            - .predict(X) -> np.ndarray of integer class labels, shape (n,)
            - .predict_proba(X) -> np.ndarray of shape (n, n_classes)
        X_val: numpy array of shape (n_val, n_features) -- standardized
        y_val: numpy array of shape (n_val,) -- integer class labels

    Returns:
        predict: callable taking X (n, n_features) and returning a numpy array
                 of integer class labels, shape (n,)
    """
    # TODO: design a combiner. Some options to consider:
    #   * Hard voting:     majority of base_models[k].predict(X) across k
    #   * Soft voting:     argmax of mean predict_proba across base models
    #   * Weighted voting: weight each model by its validation accuracy
    #   * Stacking:        train a meta-model on base predict_proba outputs
    # The win threshold is set so that naive voting strategies will not pass.
    # def _best_voter(models,X,y):
        # best_acc = 0
        # best_voter = None
        # 
        # accuracy_hv = _eval_hv(X,y,models)
        # accuracy_sv = _eval_sv(X,y,models)
        # accuracy_wv = _eval_wv(X,y,models)
        # accuracy_stacking = _eval_stacking(X,y,models)
        # 
        # acc = {
        # 'hv':accuracy_hv,
        # 'sv':accuracy_sv,
        # 'wv':accuracy_wv,
        # 'stacking':accuracy_stacking
        # }
        # for voter,accuracy in acc.items():
            # if accuracy>best_acc:
                # best_acc=accuracy
                # best_voter=voter     
        # return best_voter
    # def _eval_hv(X,y,models):
        # prediction = _predict_hv(models,X)
        # accuracy = _calc_acc(y,prediction)
        # return accuracy
    # def _eval_sv(X,y,models):
        # prediction = _predict_sv(models,X)
        # accuracy = _calc_acc(y,prediction)
        # return accuracy
    # def _eval_wv(X,y,models):
        # prediction = _predict_wv(models,X,y)
        # accuracy = _calc_acc(y,prediction)
        # return accuracy
    # def _eval_stacking(X,y,models):
        # prediction = _predict_stacking(models,X)
        # accuracy = _calc_acc(y,prediction)
        # return accuracy
    # def _predict_hv(models,X):
        # predictions =[]
        # for m in models:
            # prediction = m.predict(X)
            # predictions.append(prediction)
        # predictions=np.array(predictions)
        # return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),arr=predictions,axis=0)
    # def _predict_sv(models,X):
        # predictions =[]
        # for m in models:
            # prediction = m.predict_proba(X)
            # predictions.append(prediction)
        # predictions=np.array(predictions)
        # return np.argmax(np.mean(predictions,axis=0),axis=1)
    # def _predict_wv(models,X,y):
        # predictions=[]
        # accuracys=[]
        # for m in models:
            # prediction = m.predict_proba(X)
            # predictions.append(prediction)
            # accuracy= _calc_acc(y,np.argmax(prediction, axis=1))
            # accuracys.append(accuracy)
        # predictions=np.array(predictions)
        # accuracys=np.array(accuracys)
        # weights = accuracys/np.sum(accuracys)
        # predictions *= weights[:,None,None]
        # predictions = np.argmax(np.sum(predictions,axis=0),axis=1)
        # return predictions
    from sklearn.linear_model import LogisticRegression
    X_meta =[]
  
    for m in base_models:
        prediction = m.predict_proba(X_val)
        X_meta.append(prediction)
    X_meta =np.concatenate(X_meta,axis=1)
    meta_model = LogisticRegression()
    meta_model.fit(X_meta,y_val)
    
   

        
        
        
        
            
    
        
            
    
    def predict(X):
        # TODO: compute and return integer class predictions for X
        X_meta =[]
        for m in base_models:
            X_meta.append(m.predict_proba(X))
        
        X_meta=np.concatenate(X_meta,axis=1)
        return meta_model.predict(X_meta)
        
        # return _predict_sv(base_models,X)
        
        # raise NotImplementedError

    return predict