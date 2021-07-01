import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
import collections

# Downloaded from: https://saifmohammad.com/WebPages/nrc-vad.html
lexicon = pd.read_csv("data/NRC-VAD-Lexicon.txt", delimiter="\t")

sum_ = collections.defaultdict(list)
max_ = collections.defaultdict(list)
min_ = collections.defaultdict(list)
avg_ = collections.defaultdict(list)
std_ = collections.defaultdict(list)
range_ = collections.defaultdict(list)

def read_transcripts():
    transcription_test = pd.DataFrame.from_dict(
        pd.read_pickle("/content/drive/My Drive/phd/mood-project/p-traits/data/transcription_test.pkl"), orient='index',
        columns=['text'])
    transcription_training = pd.DataFrame.from_dict(
        pd.read_pickle("/content/drive/My Drive/phd/mood-project/p-traits/data/transcription_training.pkl"),
        orient='index', columns=['text'])
    transcription_validation = pd.DataFrame.from_dict(
        pd.read_pickle("/content/drive/My Drive/phd/mood-project/p-traits/data/transcription_validation.pkl"),
        orient='index', columns=['text'])
    transcription_test['id'] = transcription_test.index
    transcription_training['id'] = transcription_training.index
    transcription_validation['id'] = transcription_validation.index

    transcripts = []
    transcripts.append(transcription_training)
    transcripts.append(transcription_validation)
    transcripts.append(transcription_test)

    for i in [0, 1, 2]:
        transcripts[i] = transcripts[i].reset_index(drop=True)

    data = pd.concat([transcription_training, transcription_validation, transcription_test]).reset_index(drop=True)
    return data

def compute_functionals(data):
    count = 0
    for text in data['text']:
        print(".... NEW TEXT .... ", count)
        count += 1
        valence = []
        arousal = []
        dominance = []
        for word in text.split():
            score = lexicon[lexicon['Word'] == word]
            if not score.empty:
                valence.append(float(pd.to_numeric(score['Valence'])))
                arousal.append(float(pd.to_numeric(score['Arousal'])))
                dominance.append(float(pd.to_numeric(score['Dominance'])))
        if not valence:
            for feat in ['valence', 'arousal', 'dominance']:
                avg_[feat].append(0)
                sum_[feat].append(0)
                max_[feat].append(0)
                min_[feat].append(0)
                std_[feat].append(0)
                range_[feat].append(0)
        else:
            avg_['valence'].append(np.mean(valence))
            avg_['arousal'].append(np.mean(arousal))
            avg_['dominance'].append(np.mean(dominance))

            sum_['valence'].append(np.sum(valence))
            sum_['arousal'].append(np.sum(arousal))
            sum_['dominance'].append(np.sum(dominance))

            max_['valence'].append(np.amax(valence))
            max_['arousal'].append(np.amax(arousal))
            max_['dominance'].append(np.amax(dominance))

            min_['valence'].append(np.amin(valence))
            min_['arousal'].append(np.amin(arousal))
            min_['dominance'].append(np.amin(dominance))

            std_['valence'].append(np.std(valence))
            std_['arousal'].append(np.std(arousal))
            std_['dominance'].append(np.std(dominance))

            range_['valence'].append(np.amax(valence) - np.amin(valence))
            range_['arousal'].append(np.amax(arousal) - np.amin(arousal))
            range_['dominance'].append(np.amax(dominance) - np.amin(dominance))

    df = pd.DataFrame()
    for feat in ['valence', 'arousal', 'dominance']:
        df["min_" + feat] = min_[feat]
        df["max_" + feat] = max_[feat]
        df["sum_" + feat] = sum_[feat]
        df["mean_" + feat] = avg_[feat]
        df["std_" + feat] = std_[feat]
        df["range_" + feat] = range_[feat]

    df['id'] = data['id']
    df.to_csv(ROOT_PATH + "NRC-VAD-features2.csv")