import matplotlib as plt
import pickle as pkl
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


def model_inputs(real_dim, z_dim):
    """create placeholder for data
        >>> model_inputs(1, 1)

     (real_dim, z_dim): (真のデータの次元, ノイズの次元)
    """
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name="input_real")
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z")

    return inputs_real, inputs_z


def generator(z, out_dim, n_unit=128, reuse=False, alpha=0.01):
    """generate fake data using multi-NN
        >>> generator(z, 784, 128)

    n_unit: 中間層のニューロンの数
    Leaky ReLU: x < 0でゼロにしない
    reuseオプション: 関数内の変数値の保持
    """
    with tf.variable_scope("generator", reuse=reuse):
        # fully-connected
        h1 = tf.layers.dense(z, n_unit, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)
        # score, fully-connected
        logits = tf.layers.dense(h1, out_dim, activation=None)
        # [-1, 1]: 画像
        out = tf.tanh(logits)

        return out


def discriminator(x, n_unit=128, reuse=False, alpha=0.01):
    """discriminator using multi-NN
        >>> discriminator(z, 784, 128, )
    """
    with tf.variable_scope("discriminator", reuse=reuse):
        # fully-connected
        h1 = tf.layers.dense(x, n_unit, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)
        # score, fully-connected
        logits = tf.layers.dense(h1, 1, activation=None)
        # [0, 1]: 確率
        out = tf.sigmoid(logits)

        return out, logits


# MNIST_dataディレクトリにMNISTデータセットをダウンロードする
mnist = input_data.read_data_sets("MNIST_data")

"""ハイパーパラメータ"""
input_size = 784  # 入力画像サイズ 28 * 28
z_size = 100  # ランダムなベクトルサイズ
g_hidden_size = 128  # 隠れ層のノード数
d_hidden_size = 128  # 隠れ層のノード数
alpha = 0.01  # Leaky RELUの平滑度
smooth = 0.1  # 学習を円滑に進めるための値

"""モデルの定義"""
# グラフの初期化
tf.reset_default_graph()

# 入力データ定義
input_real, input_z = model_inputs(input_size, z_size)

# fake dataの作成
g_model = generator(input_z, input_size, n_unit=g_hidden_size, alpha=alpha)

# real dataの識別結果
d_model_real, d_logits_real = discriminator(
    input_real, n_unit=d_hidden_size, alpha=alpha
)
# fake dataの識別結果
d_model_fale, d_logits_fake = discriminator(
    g_model, reuse=True, n_unit=d_hidden_size, alpha=alpha
)


"""損失関数の定義"""
# 1 - smoothとの誤差, realをrealと見抜く
d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_real, labels=tf.ones_like(d_logits_real) * (1 - smooth)
    )
)
# 0との誤差, fakeをfakeと見抜く
d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_fake, labels=tf.zeros_like(d_logits_real)
    )
)

# discriminator loss
d_loss = d_loss_real + d_loss_fake

# 正解1との誤差, fakeをrealに近づけられたか
g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)
    )
)

"""最適化"""
learning_rate = 0.002

# weight, biasなどのパラメータをまとめて取り出す関数
t_vars = tf.trainable_variables()
# 分割
g_vars = [var for var in t_vars if var.name.startswith("generator")]
d_vars = [var for var in t_vars if var.name.startswith("discriminator")]

d_train_optimize = tf.train.AdamOptimizer(learning_rate).minimize(
    d_loss, var_list=d_vars
)
g_train_optimize = tf.train.AdamOptimizer(learning_rate).minimize(
    g_loss, var_list=g_vars
)


"""計算グラフの実行"""
batch_size = 100
epoch = 100
samples = []  # fakeデータの保存
losses = []  # loss per epoch

# 学習ログ保存 -> generatorを使って画像の保存
saver = tf.train.Saver(var_list=g_vars)

# session開始
with tf.Session() as sess:
    # 変数の初期化
    sess.run(tf.global_variables_initializer())
    # エポックを回す
    for e in range(epoch):
        # 1epochで学習標本全てを用いる
        for i in range(mnist.train.num_examples // batch_size):
            # 無作為抽出
            batch = mnist.train.next_batch(batch_size)
            # 784次元のベクトルに変換する
            batch_images = batch[0].reshape((batch_size, 784))
            # generatorのデータとrangeを揃える [-1, 1]
            batch_images = batch_images * 2 - 1

            # Generatorにおけるfake画像生成, 一様分布
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

            # discriminatorの最適化
            _ = sess.run(
                d_train_optimize, feed_dict={input_real: batch_images, input_z: batch_z}
            )
            # generatorの最適化
            _ = sess.run(g_train_optimize, feed_dict={input_z: batch_z})

        # loss
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})
        print(
            "エポック: {}/{}".format(e + 1, epoch),
            "D_loss: {:.4f}".format(train_loss_d),
            "G_loss: {:.4f}".format(train_loss_g),
        )

        # loss追加
        losses.append((train_loss_d, train_loss_g))

        """fake画像の作成"""
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(
            generator(
                input_z, input_size, n_unit=g_hidden_size, reuse=True, alpha=alpha
            ),
            feed_dict={input_z: sample_z},
        )
        samples.append(gen_samples)
        # モデル保存
        saver.save(sess, "./checkpoints/generator.ckpt")

# samplesのsave
with open("./train_samples.pkl", "wb") as f:
    pkl.dump(samples, f)
