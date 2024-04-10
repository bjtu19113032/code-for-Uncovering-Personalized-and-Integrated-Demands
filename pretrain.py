import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from dataloader import DataProcessor
from model import TransformerBlock,positional_encoding,BehavioralAutoencoder
class PreTrainingPipeline:
    def __init__(self, df, behavior_size=200, least_behavior=10,embedding_size=32, hidden_layer_size=64, output_size=128, epochs=20, batch_size=64, learning_rate=0.001, verbose=1):
        self.df = df
        self.behavior_size = behavior_size
        self.least_behavior = least_behavior
        self.embedding_size = embedding_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        # Initialize an empty dictionary for decoder_dims, it will be filled in process_data
        self.decoder_dims = {}

    def process_data(self):
        # Placeholder for data processing logic
        processor = DataProcessor(behavior_size=self.behavior_size, least_behavior=self.least_behavior)
        df_array = processor.process_data(self.df)
        self.raw,self.df2, self.decoder_item, self.decoder_time, self.decoder_behaviour, self.decoder_brand, self.decoder_cate = processor.prepare_data(df_array)
        
        # Define decoder dimensions based on the processed data
        self.decoder_dims = {
            'item': len(np.unique(self.decoder_item)),
            'time': len(np.unique(self.decoder_time)),
            'behaviour': len(np.unique(self.decoder_behaviour)),
            'brand': len(np.unique(self.decoder_brand)),
            'cate': len(np.unique(self.decoder_cate))
        }

    def train_model(self):
        # Initialize the BehavioralAutoencoder
        self.BAE = BehavioralAutoencoder(self.behavior_size,self.least_behavior, self.output_size, self.hidden_layer_size, self.embedding_size, self.decoder_dims)
        
        # Compile the autoencoder model
        adam = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        self.BAE.autoencoder_model.compile(optimizer=adam, loss=SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
        
        # Fit the model
        self.history = self.BAE.autoencoder_model.fit(self.df2, [self.decoder_item, self.decoder_time, self.decoder_behaviour, self.decoder_brand, self.decoder_cate], batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)

    def run(self):
        self.process_data()
        self.train_model()
        return self.BAE, self.history,self.df2, self.raw
