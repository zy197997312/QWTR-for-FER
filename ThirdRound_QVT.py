# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:38:31 2024

@author: zhouyu
"""

import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)
print(tf.__version__)

from tensorflow.keras import layers
#import tensorflow_addons as tfa
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

import numpy as np
np.random.seed(42)

# 参数设置
input_shape = (25, 25, 64)  # 修改为 (长, 宽, 通道)
num_classes = 7
learning_rate = 0.00001
batch_size = 2
num_epochs = 50
num_patches = 25 * 25  # 计算总patch数量
projection_dim = 48
num_heads = 8
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 4
mlp_head_units = [2048, 1024]


from   complexnn      import *
from tensorflow.keras.layers import (
    Dense,   
)

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = QuaternionDense(embed_dim)
        self.key_dense = QuaternionDense(embed_dim)
        self.value_dense = QuaternionDense(embed_dim)
        self.combine_heads = QuaternionDense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output

def QF_Net(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = QuaternionConv2D(int(units/4), 3, strides=1, padding="same")(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Activation(tf.nn.gelu)(x)
        x = QuaternionConv2D(int(units/4), 3, strides=1, padding="same")(x)
    return x

def multilayer_perceptron(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = QuaternionDense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        # Projection 层，用于降低每个 patch 的维度
        self.projection = layers.Dense(units=projection_dim)
        # 位置嵌入层，用于为每个 patch 添加位置编码
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        batch_size = tf.shape(patch)[0]  # 获取批次大小
        # 生成 patch 的位置序列（每个 patch 的位置）
        positions = tf.range(start=0, limit=self.num_patches, delta=1)  # 生成 [0, 1, 2, ..., 624]
        
        # 获取位置嵌入
        position_embeds = self.position_embedding(positions)  # 形状为 (625, projection_dim)
        
        # 将位置嵌入调整为 (batch_size, num_patches, projection_dim)
        position_embeds = tf.reshape(position_embeds, (1, self.num_patches, -1))  # 形状为 (1, 625, projection_dim)
        position_embeds = tf.tile(position_embeds, [batch_size, 1, 1])  # 扩展为 (batch_size, 625, projection_dim)
        
        # 对输入的 patch 进行投影，得到的形状为 (batch_size, 625, projection_dim)
        patch = tf.reshape(patch, [batch_size, -1, patch.shape[-1]])  # 将形状变为 (batch_size, 625, 64)
        encoded = self.projection(patch)  # 投影后的形状为 (batch_size, 625, projection_dim)

        # 将投影后的 patch 和位置嵌入相加
        encoded = encoded + position_embeds
        return encoded

        
def create_qvit_classifier():
    inputs = layers.Input(shape=input_shape)

    # position embedding
    encoded_patches = PatchEncoder(num_patches, projection_dim)(inputs)
    
    for _ in range(transformer_layers):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        attention_output = MultiHeadSelfAttention(projection_dim, num_heads)(x1)
        
        x2 = layers.Add()([attention_output, encoded_patches])
   
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        x4 = tf.keras.layers.Reshape((25,25,48))(x3)
       
        x5 = QF_Net(x4, hidden_units=transformer_units, dropout_rate=0.3)
        
        x6 = tf.keras.layers.Reshape((625, 48))(x5)
      
        encoded_patches = layers.Add()([x6, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = multilayer_perceptron(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    logits = layers.Dense(num_classes)(features)

    qvit_model = keras.Model(inputs=inputs, outputs=logits)
    return qvit_model

# 定义数据
data = selected_rec_images_reshaped.reshape(13671, 25, 25, 64)  # 确保形状正确
labels = np.array(labels_vectors)  # 使用实际的标签向量
label = np.argmax(labels, axis=-1)



def run_experiment(model):
    optimizer = tf.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "./RAFDB/model_{epoch:03d}-{val_accuracy:.4f}.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=data,
        y=label,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(data, label),
        callbacks=[checkpoint_callback],
    )

    return history

vit_classifier = create_qvit_classifier()
history = run_experiment(vit_classifier)






