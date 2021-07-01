### NMBR of WORDS/SENTENCES
import  nltk
import pandas as pd

def basic_statistics(X):
  word_arr = []
  sent_arr = []
  stat_arr = []
  for story in X:
    sent_arr = nltk.sent_tokenize(story)
    word_arr = nltk.word_tokenize(story)
    stat_arr.append([len(word_arr), len(sent_arr)])
  print(len(stat_arr))
  return stat_arr

if __name__ == "__main__":
    stat_arr = basic_statistics(transcription_test['text'])
    stat_features = np.asarray(stat_arr)
    df = pd.DataFrame()
    df['nmbr_word'] = stat_features[:, 0]
    df['nmbr_sent'] = stat_features[:, 1]
    df.to_csv("features/basic_stats_test.csv", index=False)
