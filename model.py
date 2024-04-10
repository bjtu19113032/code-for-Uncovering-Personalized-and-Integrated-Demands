from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf

def positional_encoding(pos, model_size):
    """Generates a positional encoding matrix for embedding temporal information."""
    PE = np.zeros((1, pos, model_size))
    for i in range(pos):
        for j in range(model_size):
            if j % 2 == 0:
                PE[:, i, j] = np.sin(i / np.power(10000, (2 * j / model_size)))
            else:
                PE[:, i, j] = np.cos(i / np.power(10000, (2 * (j - 1) / model_size)))
    return tf.constant(PE, dtype=tf.float32)

class TransformerBlock(layers.Layer):
    """Implements a Transformer block as a layer."""
    def __init__(self, embed_size, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_size)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_size),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class BehavioralAutoencoder:
    """Defines a Behavioral Autoencoder with Transformers for encoding and decoding."""
    def __init__(self, behavior_size,least_behavior, output_size, hidden_layer_size, embedding_size, decoder_dims):
        self.behavior_size = behavior_size
        self.least_behavior = least_behavior
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.embedding_size = embedding_size
        self.decoder_dims = decoder_dims

        self.encoder_model, self.item_embedding_model = self.create_encoder()
        self.decoder_model = self.create_decoder()
        self.autoencoder_model = self.create_autoencoder()

    def create_encoder(self):
        encoder_inputs = layers.Input(shape=(self.behavior_size, 5))
        item_input, timestamp_input, behavior_input, brand_input, cate_input = [encoder_inputs[:, :, i] for i in range(5)]

        embeddings = []
        for input_layer, key in zip([item_input, timestamp_input, behavior_input, brand_input, cate_input], 
                                    ['item', 'time', 'behaviour', 'brand', 'cate']):
            embedding = layers.Embedding(input_dim=self.decoder_dims[key], 
                                         output_dim=self.embedding_size, 
                                         input_length=self.behavior_size)(input_layer)
            embeddings.append(embedding)
        item_embedding=embeddings[0]
        embedding = layers.concatenate(embeddings, axis=-1)
        embedding += positional_encoding(self.behavior_size, self.embedding_size * 5)  # Correct dimension after concatenation

        x = TransformerBlock(self.embedding_size * 5, num_heads=8, ff_dim=self.hidden_layer_size)(embedding)
        encoder_outputs = layers.GlobalAveragePooling1D()(x)
        encoder_outputs = layers.Dense(self.output_size, activation='relu')(encoder_outputs)
        encoder_model=models.Model(inputs=encoder_inputs, outputs=encoder_outputs)
        item_embedding_model=models.Model(inputs=encoder_inputs, outputs=item_embedding)
        return encoder_model,item_embedding_model

    def create_decoder(self):
        decoder_inputs = layers.Input(shape=(self.output_size,))
        x = layers.RepeatVector(self.behavior_size)(decoder_inputs)
        x += positional_encoding(self.behavior_size, self.output_size)

        x = TransformerBlock(self.output_size, num_heads=8, ff_dim=self.hidden_layer_size)(x)

        item_out = layers.TimeDistributed(layers.Dense(self.decoder_dims['item'], activation='softmax'))(x)
        timestamp_out = layers.TimeDistributed(layers.Dense(self.decoder_dims['time'], activation='softmax'))(x)
        behavior_out = layers.TimeDistributed(layers.Dense(self.decoder_dims['behaviour'], activation='softmax'))(x)
        brand_out = layers.TimeDistributed(layers.Dense(self.decoder_dims['brand'], activation='softmax'))(x)
        cate_out = layers.TimeDistributed(layers.Dense(self.decoder_dims['cate'], activation='softmax'))(x)

        return models.Model(inputs=decoder_inputs, outputs=[item_out, timestamp_out, behavior_out, brand_out, cate_out])

    def create_autoencoder(self):
        encoder_inputs = layers.Input(shape=(self.behavior_size, 5))
        encoder_outputs = self.encoder_model(encoder_inputs)
        decoder_outputs = self.decoder_model(encoder_outputs)
        return models.Model(inputs=encoder_inputs, outputs=decoder_outputs)
