"""Script to train a topic model
"""
import ktrain
from Dataset import Data

data = Data(language='en', creating_parquet=False)
data = data.dataframe

tm = ktrain.text.get_topic_model(data['text'], n_features=150)
tm.print_topics()
tm.build(data['text'], threshold=0.2)
texts = tm.filter(data['text'])
categories = tm.filter(data['label'])
tm.print_topics(show_counts=True)
tm.save('text_classifier/models/english_LDA/')

