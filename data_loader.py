import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        df_item2id = pd.read_csv('data/movie/item_index2entity_id.txt', sep='\t', header=None, names=['item','id'])
        df_kg = pd.read_csv('data/movie/kg.txt', sep='\t', header=None, names=['head','relation','tail'])
        df_rating = pd.read_csv('data/movie/ratings.csv')
        
        df_rating = df_rating[df_rating['movieId'].isin(df_item2id['item'])]
        df_rating.reset_index(inplace=True)
        
        self.df_item2id = df_item2id
        self.df_kg = df_kg
        self.df_rating = df_rating
        
        self.user_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()

        self._encoding()
        
    def _encoding(self):
        self.user_encoder.fit(self.df_rating['userId'])
        self.entity_encoder.fit(pd.concat([self.df_rating['movieId'], self.df_kg['head'], self.df_kg['tail']]))
        self.relation_encoder.fit(self.df_kg['relation'])
        
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])

    def _build_dataset(self):
        print('Build dataset dataframe ...', end=' ')
        # df_rating update
        df_dataset = pd.DataFrame()
        df_dataset['userId'] = self.user_encoder.transform(self.df_rating['userId'])
        df_dataset['movieId'] = self.entity_encoder.transform(self.df_rating['movieId'])
        df_dataset['label'] = self.df_rating['rating'].apply(lambda x: 0 if x < 4.0 else 1)
        print('Done')
        return df_dataset
        
    def _construct_kg(self):
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(self.df_kg)):
            head = self.df_kg.iloc[i]['head']
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg
        
    def load_data(self):
        return self._build_dataset()

    def load_kg(self):
        return self._construct_kg()
    
    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)
    
    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))
    
    
    
    