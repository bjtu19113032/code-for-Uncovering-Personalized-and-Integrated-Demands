import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

class FeatureSelection:
    def __init__(self, BAE,embedding_size=16, behavior_size=200):
        self.embedding_size = embedding_size
        self.behavior_size = behavior_size
        self.BAE = BAE
    def calculate_item_embedding(self, df2):
        item_id_list = np.arange(0, len(np.unique(df2[:, :, 0])))
        item_id_list = np.pad(item_id_list, (0, self.behavior_size - len(item_id_list) % self.behavior_size), 'constant')
        item_id_list = item_id_list.reshape(-1, self.behavior_size)
        item_id_list = np.repeat(item_id_list, 5, axis=1)
        item_id_list = item_id_list.reshape(-1, self.behavior_size, 5)
        item_embedding = self.BAE.item_embedding_model.predict(item_id_list)
        item_embedding = item_embedding.reshape(-1, self.embedding_size)
        item_embedding = item_embedding[:len(np.unique(df2[:, :, 0])), :]
        return item_embedding

    def prepare_data(self, df2, labels):
        group_data = df2[labels == 0]
        group_data = group_data.reshape(-1, 5)
        buy = group_data[group_data[:, 1] == 4]
        unbuy = group_data[group_data[:, 1] != 4]
        buy_item_unique = np.unique(buy[:, 0])
        unbuy_item_unique = np.unique(unbuy[:, 0])
        
        item_embedding_value = self.calculate_item_embedding(df2)
        buy_embedding = item_embedding_value[buy_item_unique]
        unbuy_embedding = item_embedding_value[unbuy_item_unique]
        
        buy_embedding = pd.DataFrame(buy_embedding)
        buy_embedding['label'] = 1
        unbuy_embedding = pd.DataFrame(unbuy_embedding)
        unbuy_embedding['label'] = 0
        
        data = pd.concat([buy_embedding, unbuy_embedding], axis=0)
        return data
    
    def train_model(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)
        
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        return lr
    
    def identify_top_features(self, model):
        feature = np.zeros((2**self.embedding_size, self.embedding_size))
        for i in range(2**self.embedding_size):
            for j in range(self.embedding_size):
                feature[i, j] = (i >> j) & 1
        y_pred = model.predict_proba(feature)[:, 1]
        top_10 = np.argsort(y_pred)[::-1][:10]
        top_10_features = feature[top_10]
        return top_10_features

    def run(self, df2, labels):
        data = self.prepare_data(df2, labels)
        model = self.train_model(data)
        top_10_features = self.identify_top_features(model)
        
        for i, feature in enumerate(top_10_features):
            print(f"top{i+1}_feature: {feature}")
        
        return top_10_features
