from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from dataloader import DataProcessor
from model import TransformerBlock,positional_encoding,BehavioralAutoencoder
from loss import CustomModel
class JointTrainer:
    def __init__(self, df, BAE, n_classes=10, n_epochs=5, batch_size=32, learning_rate=0.001):
        self.df = df
        self.BAE = BAE
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = Adam(learning_rate=self.learning_rate)
        
    def run(self):
        processor = DataProcessor(behavior_size=self.BAE.behavior_size, least_behavior=self.BAE.least_behavior)
        df_array = processor.process_data(self.df)
        raw,df2, decoder_item, decoder_time, decoder_behaviour, decoder_brand, decoder_cate = processor.prepare_data(df_array)
        custom_model = CustomModel(num_behaviors=len(np.unique(decoder_behaviour)), num_categories=len(np.unique(decoder_cate)), num_products=len(np.unique(decoder_item)), n_clusters=self.n_classes)
        
        for epoch in range(self.n_epochs):
            encoded_v = []
            
            for i in range(0, len(df2), self.batch_size):
                x_batch = df2[i:i+self.batch_size]
                encoded = self.BAE.encoder_model.predict(x_batch)
                encoded_v.append(encoded)
            encoded_v = np.concatenate(encoded_v, axis=0)
            
            custom_model.update_kmeans(encoded_v)
            

            for i in range(0, len(df2), self.batch_size):
                x_batch = df2[i:i+self.batch_size]
                product_batch = decoder_item[i:i+self.batch_size]
                time_batch = decoder_time[i:i+self.batch_size]
                behaviour_batch = decoder_behaviour[i:i+self.batch_size]
                brand_batch = decoder_brand[i:i+self.batch_size]
                category_batch = decoder_cate[i:i+self.batch_size]

                with tf.GradientTape() as tape:
                    reconstructed = self.BAE.autoencoder_model(x_batch)
                    loss = custom_model.custom_loss(product_batch, reconstructed[0], time_batch, reconstructed[1], behaviour_batch, reconstructed[2], brand_batch, reconstructed[3], category_batch, reconstructed[4], encoded_v)

                gradients = tape.gradient(loss, self.BAE.autoencoder_model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.BAE.autoencoder_model.trainable_variables))

            # Calculate clustering metrics
            cluster_labels = custom_model.kmeans.labels_
            silhouette_avg = silhouette_score(encoded_v, cluster_labels)
            calinski_harabasz_score_val = calinski_harabasz_score(encoded_v, cluster_labels)
            print(f'Epoch {epoch}, Loss: {loss.numpy():.4f}, SS: {silhouette_avg:.4f}, CH: {calinski_harabasz_score_val:.4f}')
        return cluster_labels 