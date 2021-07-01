import flair
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
nltk.download('punkt')
import ast

sid = []
flair_sentiment = []

def read_data(filename, delim):
  df = pd.read_csv(filename, delimiter=delim)
  return df

def load_analyzer():
  nltk.download('vader_lexicon')
  sid = SentimentIntensityAnalyzer()
  flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
  return (sid, flair_sentiment)

(sid, flair_sentiment) = load_analyzer()

def nltk_get_sentiment(sentence):
    return sid.polarity_scores(sentence)

def textblob_get_sentiment(sentence):
    # we can use subjectivity analysis from textblob result since polarity prediction is not as good as NLTK.
    return TextBlob(sentence).sentiment

def flair_get_sentiment(sentence):
    s = flair.data.Sentence(sentence)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    s = ' '.join(map(str, total_sentiment))
    prob = s[s.find("(") + 1:s.find(")")]
    if "NEGATIVE" in s:
        prob = 1-abs(float(prob))
    return (prob)

# Polarity Feature Extraction per Story
def write_sentiment_features(df, filename):
    df_final = pd.DataFrame()
    sent_arr = get_sentiment_features_per_story(df)
    polarity_features = np.asarray(sent_arr)
    df_final["polarity"] = polarity_features[:, 0]
    df_final["pos"] = polarity_features[:, 1]
    df_final["neg"] = polarity_features[:, 2]
    df_final["neu"] = polarity_features[:, 3]
    df_final["compound"] = polarity_features[:, 4]
    df_final["subjectivity"] = polarity_features[:, 5]
    df_final["flair_prob"] = polarity_features[:, 6]
    df_final["flair_label"] = polarity_features[:, 7]

    df_final.to_csv(filename+"polarity_features.csv", index=False)

def compute_all_sent_features(text):
    sentiment = nltk_get_sentiment(text)
    textblob_sent = textblob_get_sentiment(text)
    (flair_prob) = flair_get_sentiment(text)
    return ([textblob_sent[0], sentiment["pos"], sentiment["neg"],
             sentiment["neu"], sentiment["compound"], textblob_sent[1], flair_prob])

def get_sentiment_features_per_story(X):
    story_arr = []
    i = 0
    for story in X:
        if story is not None:
          story_arr.append(compute_all_sent_features(story))
          i = i + 1
        else:
            story_arr.append([0, 0, 0, 0, 0, 0, 0, 0])
    return story_arr

def get_sentiment_features_per_sentence(X):
    story_arr = []
    flair_arr = []
    label_arr = []
    textblob_polarity = []
    pos = []
    neg =[]
    neu = []
    compound = []
    textblob_subject = []
    i = 0
    for story in X:
        for sent in nltk.sent_tokenize(story):
            (flair_prob, flair_label) = flair_get_sentiment(sent)
            sentiment = nltk_get_sentiment(sent)
            textblob_sent = textblob_get_sentiment(sent)
            textblob_polarity.append(textblob_sent[0])
            textblob_subject.append(textblob_sent[1])
            pos.append(sentiment["pos"])
            neg.append(sentiment["neg"])
            neu.append(sentiment["neu"])
            compound.append(sentiment["compound"])
            flair_arr.append(flair_prob)
            label_arr.append(flair_label)
        story_arr.append([textblob_polarity, pos, neg, neu, compound, textblob_subject, flair_arr, label_arr])
        flair_arr = []
        label_arr = []
        pos = []
        neg = []
        neu = []
        compound = []
        textblob_subject = []
        textblob_polarity = []
        i = i + 1
    return story_arr

### Summary functions for sentence-level data

def to_list(column_name):
    df[column_name] = df[column_name].apply(ast.literal_eval)

def unfold(key_column_name, value_column_name):
    df_unfold[[key_column_name, value_column_name]] = df[[key_column_name, value_column_name]].explode(value_column_name)[[key_column_name, value_column_name]]

def write_summary_functions(df_unfold, split):
  min_max_df = df_unfold.groupby(filename_column).agg(min_max)
  min_max_df.columns = ['_'.join(col) for col in min_max_df.columns]
  min_max_df = min_max_df.reset_index()
  min_max_df.to_csv('/content/drive/My Drive/phd/CSL-models/Cha-learn-annotations/polarity/sum_personality_traits_per_sentence_'+split+".csv", index=False, header=True)

if __name__ == "__main__":
    [to_list(x) for x in df.set_index([filename_column]).keys().tolist()]
    df_unfold = pd.DataFrame()
    [unfold(filename_column, x) for x in df.set_index([filename_column]).keys().tolist()]
    df_unfold['flair_prob'] = df_unfold['flair_prob'].astype('float32')
    df_unfold['polarity'] = df_unfold['polarity'].astype('float32')
    df_unfold['neg'] = df_unfold['neg'].astype('float32')
    df_unfold['pos'] = df_unfold['pos'].astype('float32')
    df_unfold['compound'] = df_unfold['compound'].astype('float32')
    df_unfold['neu'] = df_unfold['neu'].astype('float32')
    df_unfold['subjectivity'] = df_unfold['subjectivity'].astype('float32')
    df_unfold['rank'] = df_unfold.groupby([filename_column]).cumcount() + 1
    df_unfold['flair_prob'] = np.where(df_unfold['flair_label'] == 'negative', 1.0 - df_unfold['flair_prob'],
                                       df_unfold['flair_prob'])
    df_unfold.drop(['flair_label'], axis=1, inplace=True)

    min_max = {}

    min_max.update({'flair_prob': ['min', 'max', 'mean', 'std', 'sum']})
    min_max.update({'polarity': ['min', 'max', 'mean', 'std', 'sum']})
    min_max.update({'neg': ['min', 'max', 'mean', 'std', 'sum']})
    min_max.update({'pos': ['min', 'max', 'mean', 'std', 'sum']})
    min_max.update({'compound': ['min', 'max', 'mean', 'std', 'sum']})
    min_max.update({'neu': ['min', 'max', 'mean', 'std', 'sum']})
    min_max.update({'subjectivity': ['min', 'max', 'mean', 'std', 'sum']})

    write_summary_functions(df_unfold, "validation")