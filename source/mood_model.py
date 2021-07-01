from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import recall_score
from collections import Counter
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import sklearn.metrics
import sys
from sklearn.decomposition import PCA
np.set_printoptions(threshold=sys.maxsize)
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def evaluate(y, y_pred):
    # Evaluation
    return (recall_score(y, y_pred, average='macro')* 100)

ROOT_PATH = "/content/drive/My Drive/phd/mood-project/"

mood_begin = 660
mood_end = 960


def update_y(class_type, label):
    annots_ = pd.read_csv(ROOT_PATH + 'mood_annots/anno_gt_' + class_type + '.csv')[['Filename', label]]
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_train = annots_.iloc[:mood_begin][label]
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_blind = annots_.iloc[mood_begin:mood_end][label]
    y_blind = le.transform(y_blind)
    annots_ = annots_.rename(columns={"Filename": "id"})
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


def experiment(features):
    # feature-level fusion for at most 5 different features
    y = np.ravel(y_train)
    print("Length of training data {t} and test data {tt}".format(t=features[0].shape, tt=features[1].shape))
    clf = tune(features[0].values, y)
    y_preds = clf.predict(features[1].values)
    dev_UAR = clf.best_score_
    # print("EVALUATION SCORE UAR%:({}) ".format(evaluate(y_blind, y_preds)))
    return (evaluate(y_blind, y_preds), dev_UAR, clf)

def read_aco_features():
    ###. Read Audio/Video PCA features
    ### TRAIT ARR ###
    trait_arr = ["openness", "agreeableness", "conscientiousness", "extraversion", "neuroticism", "interview"]
    p_traits_annots = []
    for split in ["training", "validation", "test"]:
        print("split : ", split)
        p_traits_annots.append(
            pd.DataFrame.from_dict(pd.read_pickle(ROOT_PATH + "p-traits/data/annotation_" + split + ".pkl")))
        p_traits_annots[-1]['id'] = p_traits_annots[-1].index
        p_traits_annots[-1] = p_traits_annots[-1].reset_index(drop=True)

    audio = pd.read_csv(ROOT_PATH + "audio_pca.csv", index_col=0)
    video = pd.read_csv(ROOT_PATH + "video_pca.csv", index_col=0)

    audio_pca = [audio[0:6000].reset_index(drop=True), audio[6000:8000].reset_index(drop=True),
                 audio[8000: 10000].reset_index(drop=True)]
    video_pca = [video[0:6000].reset_index(drop=True), video[6000:8000].reset_index(drop=True),
                 video[8000: 10000].reset_index(drop=True)]

    for i in [0, 1, 2]:
        audio_pca[i] = pd.concat([p_traits_annots[i]['id'], audio_pca[i]], axis=1)
        video_pca[i] = pd.concat([p_traits_annots[i]['id'], video_pca[i]], axis=1)


def read_ling_features():
    summary = []
    stats = []
    LIWC = []
    bert = []
    tfidf = []
    fasttext = []
    polarity = []
    lda = []
    rep = []
    valence = []
    arousal = []
    like = []
    i = 0
    for split in ["training", "validation", "test"]:
        tfidf.append(pd.read_csv('features/tfidf_' + split + '.csv'))
        fasttext.append(pd.read_csv(ROOT_PATH + 'features/fasttext_' + split + '.csv'))
        polarity.append(pd.read_csv(ROOT_PATH + 'features/polarity_' + split + '.csv'))
        # not in the same order with the transcriptions.
        summary.append(pd.read_csv(ROOT_PATH + 'features/summary_' + split + '.csv').fillna(0))
        # all in the same order with the transcripts data set..
        stats.append(pd.read_csv(ROOT_PATH + 'features/stats_' + split + '.csv'))
        LIWC.append(pd.read_csv(ROOT_PATH + 'features/LIWC_' + split + '.csv'))
        bert.append(pd.read_csv(ROOT_PATH + 'features/bert_' + split + '.csv'))
        lda.append(pd.read_csv(ROOT_PATH + "p-traits/lda_results_" + split + '.csv', index_col=0))
        rep.append(pd.read_csv(ROOT_PATH + "features/rep_" + split + ".csv", index_col=0))
        i += 1

    VAD = pd.read_csv((ROOT_PATH + "NRC-VAD-features.csv"), index_col=0)
    vad = [VAD[0:6000], VAD[6000:8000], VAD[8000:10000]]

    VAD2 = pd.read_csv((ROOT_PATH + "NRC-VAD-features2.csv"), index_col=0)
    vad2 = [VAD2[0:6000], VAD2[6000:8000], VAD2[8000:10000]]

    for i in [0, 1, 2]:
        for col in lda[i].columns:
            lda[i][col] = lda[i][col].apply(lambda x: float(x.replace(")", "").split(",")[1].strip()))
        lda[i] = pd.concat([LIWC[i]['id'], lda[i]], axis=1)


if __name__ == "__main__":
    label = "GT_base"
    class_type = "valence"
    (mood_annots_, y_train, y_blind) = update_y(class_type, label)

    annots_all = pd.read_csv(ROOT_PATH + "mood_annots/mood_annots.csv", index_col=0)
    from itertools import combinations

    # audio = get_mood_subset_for_training(audio[0])
    audio_pca_ordered = get_mood_subset_for_training(audio_pca[0])

    # video = get_mood_subset_for_training(video[0])
    video_pca_ordered = get_mood_subset_for_training(video_pca[0])
    from itertools import combinations

    summary_ordered = get_mood_subset_for_training(summary[0])
    LIWC_ordered = get_mood_subset_for_training(LIWC[0])
    bert_ordered = get_mood_subset_for_training(bert[0])
    stats_ordered = get_mood_subset_for_training(stats[0])
    rep_ordered = get_mood_subset_for_training(rep[0])
    vad_ordered = get_mood_subset_for_training(vad[0])
    tfidf_ordered = get_mood_subset_for_training(tfidf[0])

    # scene_pca = get_mood_subset_for_training(scene_pca[0])