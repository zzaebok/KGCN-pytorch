import pandas as pd
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
        
    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)
    
    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_)
        
    def encoding(self):
        users = np.array(list(set(df_rating['userId'])))
        entities = np.array(list(set(df_rating_final['movieId']) | set(df_kg['head']) | set(df_kg['tail'])))
        relations = np.array(list(set(df_kg['relation'])))

        self.user_encoder.fit(users)
        self.entity_encoder.fit(entities)
        self.relation_encoder.fit(relations)
        
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])
        
    def construct_kg(self):
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
        self.kg = kg
        
    def build_dataset(self):
        # df_rating update
        df_dataset = pd.DataFrame()
        df_dataset['userId'] = self.user_encoder.transform(self.df_rating['userId'])
        df_dataset['movieId'] = self.user_encoder.transform(self.df_rating['movieId'])
        df_dataset['label'] = self.df_rating['rating'].apply(lambda x: 0 if x < 4.0 else 1)
        self.df_dataset = df_dataset
        
    
    def load_data():
        return train_test_split(self.df_dataset, self.df_dataset['label'], test_size=0.2,train_size=0.8)
    
    
    
    
    