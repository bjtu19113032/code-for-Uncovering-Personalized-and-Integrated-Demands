import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

class CustomModel:
    def __init__(self, num_behaviors, num_categories, num_products, n_clusters=10, alpha=0.1, beta=0.1, outlier_threshold=10, threshold=2):
        self.num_behaviors = num_behaviors
        self.num_categories = num_categories
        self.num_products = num_products
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.outlier_threshold = outlier_threshold
        self.threshold = threshold
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)

    def update_kmeans(self, encoded_data):
        self.kmeans.fit(encoded_data)
        
    def re_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        return loss_fn(y_true, y_pred)

    def extract_features(self, sequence):
        sequence = tf.cast(sequence, dtype=tf.int32)
        behavior_freq = tf.math.bincount(sequence[:, 0], minlength=self.num_behaviors, maxlength=self.num_behaviors, dtype=tf.float32)
        category_freq = tf.math.bincount(sequence[:, 1], minlength=self.num_categories, maxlength=self.num_categories, dtype=tf.float32)
        product_freq = tf.math.bincount(sequence[:, 2], minlength=self.num_products, maxlength=self.num_products, dtype=tf.float32)
        behavior_freq = behavior_freq / tf.reduce_sum(behavior_freq)
        category_freq = category_freq / tf.reduce_sum(category_freq)
        product_freq = product_freq / tf.reduce_sum(product_freq)
        return tf.concat([behavior_freq, category_freq, product_freq], axis=0)

    def preference_loss(self, sequence1, sequence2):
        features1 = self.extract_features(sequence1)
        features2 = self.extract_features(sequence2)
        return tf.norm(features1 - features2)

    def custom_loss(self, item_true, item_pred, time_true, time_pred, behavior_true, behavior_pred, brand_true, brand_pred, cate_true, cate_pred, encoded):
        item_loss = self.re_loss(item_true, item_pred)
        time_loss = self.re_loss(time_true, time_pred)
        behaviour_loss = self.re_loss(behavior_true, behavior_pred)
        brand_loss = self.re_loss(brand_true, brand_pred)
        cate_loss = self.re_loss(cate_true, cate_pred)

        encoded_np = np.array(encoded)
        cluster_assignments = self.kmeans.predict(encoded_np)
        cluster_assignments = tf.cast(cluster_assignments, tf.int32)
        cluster_centers = tf.convert_to_tensor(self.kmeans.cluster_centers_, dtype=tf.float32)
        difference = encoded - tf.gather(cluster_centers, cluster_assignments)
        distances = tf.norm(difference, axis=1)
        mask = distances < self.outlier_threshold
        filtered_distances = tf.boolean_mask(distances, mask)
        clustering_loss = tf.reduce_mean(filtered_distances)

        pred = tf.stack([tf.argmax(behavior_pred, axis=2), tf.argmax(cate_pred, axis=2), tf.argmax(item_pred, axis=2)], axis=2)
        true = tf.stack([behavior_true, cate_true, item_true], axis=2)
        prefer_loss = self.preference_loss(pred, true)

        return (time_loss + behaviour_loss + item_loss + brand_loss + cate_loss) + self.alpha * clustering_loss + self.beta * prefer_loss

    def remove_outliers(self, encoded_data, cluster_predictions):
        distances = np.linalg.norm(encoded_data - self.kmeans.cluster_centers_[cluster_predictions], axis=1)
        mask = distances < self.threshold
        return encoded_data[mask], cluster_predictions[mask]
