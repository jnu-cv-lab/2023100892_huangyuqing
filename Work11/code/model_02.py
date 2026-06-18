import tensorflow as tf
from tensorflow.keras import layers

def SkeletonTransformer(seq_len, feature_dim, num_classes):
    # 输入层
    inputs = layers.Input(shape=(seq_len, feature_dim))

    # 特征投影
    x = layers.Dense(128)(inputs)

    # 位置编码
    pos_indices = tf.range(0, seq_len)
    pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=128)(pos_indices)
    x = x + pos_embedding

    # 多头自注意力
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)
    attn_out = attn(query=x, key=x, value=x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

    # FFN前馈网络
    ffn = tf.keras.Sequential([
        layers.Dense(256, activation="relu"),
        layers.Dense(128)
    ])
    ffn_out = ffn(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)

    # 全局池化 + 分类输出
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="skeleton_transformer")