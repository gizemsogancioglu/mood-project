#%pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer
import pandas as pd
model = SentenceTransformer('bert-base-nli-mean-tokens')
#model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

sentences_training =pd.read_csv("/content/drive/My Drive/phd/mood-project/p-traits/data/transcription_training_.csv")['text']
sentences_validation =pd.read_csv("/content/drive/My Drive/phd/mood-project/p-traits/data/transcription_validation_.csv")['text']
sentences_test =pd.read_csv("/content/drive/My Drive/phd/mood-project/p-traits/data/transcription_test_.csv")['text']
sentences = pd.concat([sentences_training, sentences_validation, sentences_test])
sentences = sentences.apply(str).to_list()
sentence_embeddings = model.encode(sentences)

bert = []
bert.append(pd.DataFrame(sentence_embeddings[0:6000]))
bert.append(pd.DataFrame(sentence_embeddings[6000:8000]))
bert.append(pd.DataFrame(sentence_embeddings[8000:10000]))

bert[0].to_csv("/content/drive/My Drive/phd/mood-project/p-traits/bert/bert_training.csv")
bert[1].to_csv("/content/drive/My Drive/phd/mood-project/p-traits/bert/bert_validation.csv")
bert[2].to_csv("/content/drive/My Drive/phd/mood-project/p-traits/bert/bert_test.csv")


