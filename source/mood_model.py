import collections
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
np.set_printoptions(threshold=sys.maxsize)
import os
svm_params = [{
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['sigmoid', 'rbf'],
        'class_weight': ["balanced"],
        'probability': [True]
    }]

def evaluate(y, y_pred):
    # Evaluation
    return (recall_score(y, y_pred, average='macro')* 100)

mood_begin = 660
mood_end = 960
class_dimensions = ['valence', 'arousal', 'likeability']

def read_annots():
    annots_ = pd.read_csv('../data/mood_annots.csv')
    y_train = annots_.iloc[:mood_begin]
    y_blind = annots_.iloc[mood_begin:mood_end]
    le = preprocessing.LabelEncoder()

    for class_ in class_dimensions:
        y_train[class_] = le.fit_transform((y_train[class_]))
        y_blind[class_] = le.transform((y_blind[class_]))

    return (annots_, pd.DataFrame(y_train), pd.DataFrame(y_blind))


def get_mood_subset_for_training(data):
    ordered_data = []
    train_ids = mood_annots_['id'][0:mood_begin]
    test_ids = mood_annots_['id'][mood_begin:mood_end]
    for ids in [train_ids, test_ids]:
        ordered_data.append(data.loc[data['id'].isin(ids)])
        ordered_data[-1] = (pd.DataFrame(mood_annots_['id']).merge(ordered_data[-1], on='id')).drop(
            columns=['id']).reset_index(drop=True)
    return ordered_data

def tune(X_train, y_train):
    grid = GridSearchCV(
       SVC(), svm_params, scoring='%s_macro' % "recall", verbose=1, cv=3)
    grid.fit(X_train, y_train)
    return grid

def experiment(mood_feat):
    multimodal_ff = early_fusion(mood_feat)
    for mood in ['arousal']:
        grid = tune(multimodal_ff[mood][0], np.ravel(y_train[mood]))
        print("CV score of model {model}: {mood} ".format(model=mood, mood=grid.best_score_))
        test_preds = grid.predict(multimodal_ff[mood][1])
        test_score = evaluate(np.ravel(y_blind[mood]), test_preds)
        #print(test_preds)
        print("Test set score of {model} model is : {score}".format(model=mood, score=test_score))
    return

def early_fusion(mood_feat):
    # Valence model : [Polarity, TEPS, VGG-FER]
    # Arousal model: [VAD, IS13, VGG-FER]
    # Likeability model : [TEPS, VAD, IS13, VGG-FER]
    l1_model = collections.defaultdict(list)
    for class_ in class_dimensions:
        (mood_feat['video_' + class_], l1_model['video_' + class_]) = l1_feature_selection(transform_before_fusion(mood_feat['video'], 'sc'), y_train[class_])
        (mood_feat['audio_' + class_], l1_model['video_' + class_]) = l1_feature_selection(transform_before_fusion(mood_feat['audio'], 'minmax'), y_train[class_])

    multimodal_ff = collections.defaultdict(list)
    for i in [0, 1]:
        multimodal_ff['valence'].append(pd.concat([mood_feat['video_valence'][i], mood_feat['summary'][i], mood_feat['stats'][i]], axis=1))
        multimodal_ff['arousal'].append(
            pd.concat([mood_feat['vad'][i], mood_feat['video_arousal'][i], mood_feat['audio_arousal'][i]], axis=1))
        multimodal_ff['likeability'].append(
            pd.concat([mood_feat['vad'][i], mood_feat['stats'][i], mood_feat['video_likeability'][i], mood_feat['audio_likeability'][i]], axis=1))

    return multimodal_ff

def read_sensor_features():
    ###. Read Audio/Video PCA features
    p_traits_annots = []
    for split in ["training", "validation", "test"]:
        p_traits_annots.append(
            pd.DataFrame.from_dict(pd.read_pickle("../data/p-traits/annotation_" + split + ".pkl")))
        p_traits_annots[-1]['id'] = p_traits_annots[-1].index
        p_traits_annots[-1] = p_traits_annots[-1].reset_index(drop=True)

    audio = pd.read_csv("../features/audio_pca.csv", index_col=0)
    video = pd.read_csv("../features/video_pca.csv", index_col=0)

    audio_pca = [audio[0:6000].reset_index(drop=True), audio[6000:8000].reset_index(drop=True),
                 audio[8000: 10000].reset_index(drop=True)]
    video_pca = [video[0:6000].reset_index(drop=True), video[6000:8000].reset_index(drop=True),
                 video[8000: 10000].reset_index(drop=True)]

    for i in [0, 1, 2]:
        audio_pca[i] = pd.concat([p_traits_annots[i]['id'], audio_pca[i]], axis=1)
        video_pca[i] = pd.concat([p_traits_annots[i]['id'], video_pca[i]], axis=1)

    return (audio_pca, video_pca)

def read_ling_features():
    ling_feat = collections.defaultdict(list)
    for ling in ['tfidf', 'summary', 'stats', 'LIWC', 'bert', 'vad']:
        for split in ["training", "validation", "test"]:
            ling_feat[ling].append(pd.read_csv('../features/' + ling + '_' + split + '.csv'))
            if ling == "summary":
                ling_feat['summary'][-1] = ling_feat['summary'][-1].fillna(0)

    return ling_feat

def transform_before_fusion(data, scaler):
  transformed_data = []
  if scaler == "minmax":
    scaler = MinMaxScaler()
  else:
    scaler = StandardScaler()
  transformed_data.append(pd.DataFrame(scaler.fit_transform(data[0])))
  for i in range(len(data)-1):
    transformed_data.append(pd.DataFrame(scaler.transform(data[i+1])))
  return transformed_data

def l1_feature_selection(data, y):
    transformed_data = []
    clf = LinearSVC(C=0.1, penalty="l1", dual=False, class_weight="balanced").fit(data[0], y)
    l1_model = SelectFromModel(clf, prefit=True)
    for i in range(len(data)):
        transformed_data.append(pd.DataFrame(l1_model.transform(data[i])))
    return (transformed_data, l1_model)

if __name__ == "__main__":
    trait_arr = ["openness", "agreeableness", "conscientiousness", "extraversion", "neuroticism", "interview"]
    mood_feat = {}
    (mood_annots_, y_train, y_blind) = read_annots()
    y = [y_train, y_blind]

    if os.path.isfile('../predictions/valence_preds_training.csv'):
        print("Mood+likeability predictions are already in the following folder : (predictions/valence_predictions.csv). Skipping training of first-level mood classifier.. ")
    else:
        (audio_pca, video_pca) = read_sensor_features()
        ling_feat = read_ling_features()

        for ling in ling_feat.keys():
            mood_feat[ling] = get_mood_subset_for_training(ling_feat[ling][0])
        mood_feat['audio'] = get_mood_subset_for_training(audio_pca[0])
        mood_feat['video'] = get_mood_subset_for_training(video_pca[0])

        experiment(mood_feat)