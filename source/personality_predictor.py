import collections
from sklearn.metrics import mean_absolute_error
from interpret.glassbox import ExplainableBoostingRegressor, LinearRegression
import pandas as pd
from source import mood_model

def evaluate(y, y_pred):
   return ((mean_absolute_error(y, y_pred, multioutput='raw_values')))

def ebm_regression(features):
    mean = 0
    x = pd.concat([features[0], features[1]]).reset_index(drop=True)
    feat_importances = None
    for trait in trait_arr:
        y_train = pd.concat([y[0], y[1]]).reset_index(drop=True)
        ebm = ExplainableBoostingRegressor(validation_size=0.25, random_state=1, n_jobs=-1)
        ebm.fit(x, y_train[trait])
        y_preds = ebm.predict(features[2])
        print(evaluate(y[2][trait], y_preds))
        mean += evaluate(y[2][trait], y_preds)
        print("Feature importances for trait {trait}".format(trait=ebm.feature_importances_))
        if feat_importances is not None:
            feat_importances = pd.concat([(feat_importances), pd.Series(ebm.feature_importances_)], axis=1)
        else:
            feat_importances = pd.Series(ebm.feature_importances_)
    feat_importances.columns = trait_arr
    feat_importances.index = ebm.feature_names
    feat_importances.to_csv("../predictions/model_feat_importances.csv")
    print("MEAN SCORE of Big-5 predictions in the test set is : {score}".format(score= mean/5))
    print("Explanations (feature importances of model) are saved in the : predictions/model_feat_importances.csv")
    return ebm

def read_annots():
    p_traits_annots = []
    for split in ["training", "validation", "test"]:
        p_traits_annots.append(
            pd.DataFrame.from_dict(pd.read_pickle("../data/p-traits/annotation_" + split + ".pkl")))
        p_traits_annots[-1]['id'] = p_traits_annots[-1].index
        p_traits_annots[-1] = p_traits_annots[-1].reset_index(drop=True)
    return p_traits_annots

def read_mood_preds():
    mood_preds = collections.defaultdict(list)
    for t in mood_model.class_dimensions:
        for split in ['training', 'validation', 'test']:
            mood_preds[t].append(pd.read_csv("../predictions/"+ t + "_preds_" + split + ".csv").rename(columns={'0': 'low_'+t, '1': 'medium_'+t, '2': 'high_'+t}))
    return mood_preds

if __name__ == "__main__":
    trait_arr = ["openness", "agreeableness", "conscientiousness", "extraversion", "neuroticism"]
    annotations = read_annots()
    y = [item[trait_arr] for item in annotations]

    mood_preds = read_mood_preds()
    features = []
    for i in [0, 1, 2]:
        features.append(pd.concat([mood_preds['valence'][i], mood_preds['arousal'][i], mood_preds['likeability'][i]], axis=1).drop(columns=['Unnamed: 0']))
    ebm_regression(features)
