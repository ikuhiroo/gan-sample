import matplotlib.pyplot as plt
import pickle as pkl
import tensorflow as tf
import numpy as np
# matlabのファイルを扱うためのライブラリ
from scipy.io import loadmat

def scale(x, feature_ranges=(-1, 1)):
    """データの正規化"""
    # [0, 255] -> [0, 1]
    x = (x - x.min()) / (255 - x.min())
    min, max = feature_ranges
    # [0, 1] -> [-1, 1]
    x = x * (max - min) + min
    return x

def model_inputs(real_dim, z_dim):
    """変数（プレースホルダ）を初期化する関数"""
    # 変動するマーク *
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name="input_real")
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z")

    return inputs_real, inputs_z


def generator(z, output_dim, reuse=False, alpha=0.2, training=True):
    """generate fake data using CNN"
    deconvolution: 畳み込み（圧縮）と逆の処理, データを大きくする
    """
    with tf.variable_scope("generator", reuse=reuse):
        # fully-connected
        x1 = tf.layers.dense(z, 4 * 4 * 512)
        # 1次元にreshape, 可変長の4*4*512
        x1 = tf.reshape(x1, (-1, 4, 4, 512))
        # データの偏りを調整する処理
        x1 = tf.layers.batch_normalization(x1, training=training)
        x1 = tf.maximum(alpha * x1, x1)
        
        # 畳み込み
        x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2, padding="same")
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.maximum(alpha * x2, x2)
        # -> 8 * 8 * 256

        x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2, padding="same")
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf.maximum(alpha * x3, x3)
        # -> 16 * 16 * 128

        # score, fully-connected
        logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2, padding="same")
        # -> 32 * 32 * 3
        
        # [-1, 1]: 画像
        out = tf.tanh(logits)

        return out

def discriminator(x, reuse=False, alpha=0.2):
    """discriminator using CNN
    """
    with tf.variable_scope("discriminator", reuse=reuse):
        # 32 * 32 * 3
        # 畳み込み
        x1 = tf.layers.conv2d(x, 64, 5, strides=2, padding="same")
        x1 = tf.maximum(alpha * x1, x1)
        # -> 16 * 16 * 64

        x2 = tf.layers.conv2d(x1, 128, 5, strides=2, padding="same")
        x2 = tf.layers.batch_normalization(x2, training=True)
        x2 = tf.maximum(alpha * x2, x2)
        # -> 8 * 8 * 128
        
        x3 = tf.layers.conv2d(x2, 256, 5, strides=2, padding="same")
        x3 = tf.layers.batch_normalization(x3, training=True)
        x3 = tf.maximum(alpha * x3, x3)
        # -> 4 * 4 * 256

        # score, fully-connected
        flat = tf.reshape(x3, (-1, 4*4*256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

        return out, logits

def model_loss(input_real, input_z, output_dim, alpha=0.2):
    """損失関数の定義"""
    # fake dataの作成
    g_model = generator(input_z, output_dim, alpha=alpha)

    # real dataの識別結果
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)

    # fake dataの識別結果
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, alpha=alpha)

    # 1 - smoothとの誤差, realをrealと見抜く
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_real, labels=tf.ones_like(d_model_real)
        )
    )

    # 0との誤差, fakeをfakeと見抜く
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)
        )
    )

    # discriminator loss
    d_loss = d_loss_real + d_loss_fake

    # 正解1との誤差, fakeをrealに近づけられたか
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_fake, labels=tf.ones_like(d_model_fake)
        )
    )
    
    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """最適化"""
    # weight, biasなどのパラメータをまとめて取り出す関数
    t_vars = tf.trainable_variables()
    # 分割
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
            d_loss, var_list=d_vars
        )
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
            g_loss, var_list=g_vars
        )
    return d_train_opt, g_train_opt

def view_samples(epoch, samples, nrows, ncols, figsize=(5, 5)):
    """イメージに変換して表示する"""
    # 画像と軸を描画する
    fig, axes = plt.subplots(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        sharey=True,
        sharex=True,
        facecolor="w",
    )
    for ax, img in zip(axes.flatten(), samples[epoch]):
        # 軸の設定
        ax.axis("off")
        # 正規化
        img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
        # 四角形
        ax.set_adjustable("box-forced")
        # 縦横比が同じ
        im = ax.imshow(img, aspect="equal")

    # 余白の設定
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig, axes

def train(net, dataset, epochs, batch_size, print_every=10, show_every=100, figsize=(5, 5)):
    saver = tf.train.Saver()
    sample_z = np.random.uniform(-1, 1, size=(72, z_size))

    samples, losses = [], []
    steps = 0

    # session開始
    with tf.Session() as sess:
        # 変数の初期化
        sess.run(tf.global_variables_initializer())
        # エポックを回す
        for e in range(epochs):
            # 1epochで学習標本全てを用いる
            for x, y in dataset.batches(batch_size):
                steps += 1

                # Generatorにおけるfake画像生成, 一様分布
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

                # discriminatorの最適化
                _ = sess.run(
                    net.d_opt, feed_dict={net.input_real: x, net.input_z: batch_z}
                )
                # generatorの最適化
                _ = sess.run(
                    net.g_opt, feed_dict={net.input_z: batch_z, net.input_real: x}
                )

                # loss
                if steps % print_every == 0:
                    train_loss_d = net.d_loss.eval(
                        {net.input_z: batch_z, net.input_real: x}
                    )
                    train_loss_g = net.g_loss.eval({net.input_z: batch_z})
                    print(
                        "エポック: {}/{}".format(e + 1, epochs),
                        "D_loss: {:.4f}".format(train_loss_d),
                        "G_loss: {:.4f}".format(train_loss_g),
                    )

                    # loss追加
                    losses.append((train_loss_d, train_loss_g))

                """fake画像の作成"""
                if steps % show_every == 0:
                    gen_samples = sess.run(
                        generator(net.input_z, 3, reuse=True, training=False),
                        feed_dict={net.input_z: sample_z},
                    )
                    samples.append(gen_samples)
                    _ = view_samples(-1, samples, 6, 12, figsize=figsize)
                    plt.show()

        # モデル保存
        saver.save(sess, "./checkpoints/generator.ckpt")

    # samplesのsave
    with open("./samples.pkl", "wb") as f:
        pkl.dump(samples, f)

    return losses, samples

# データセットのクラスを定義する
class Dataset:
    """
    val_frac: test dataをtrainとvalで分離する割合
    scale_func: データのスケールを別に用意するか
    """

    def __init__(self, train, test, val_frac=0.5, shuffle=False, scale_func=None):
        # test dataをtestとvalに分割する
        split_index = int(len(test["y"]) * (1 - val_frac))
        # test用とval用の入力データ
        self.test_x, self.valid_x = (
            test["X"][:, :, :, :split_index],
            test["X"][:, :, :, split_index:],
        )
        # test用とval用の教師データ
        self.test_y, self.valid_y = (
            test["y"][:split_index],
            test["y"][split_index:],
        )
        # train用の入力データと教師データ
        self.train_x, self.train_y = train["X"], train["y"]

        # Tensorflow形式に変換: {index, R, G, B}
        self.train_x = np.rollaxis(self.train_x, 3)
        self.valid_x = np.rollaxis(self.valid_x, 3)
        self.test_x = np.rollaxis(self.test_x, 3)

        # データ正規化関数の指定
        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func

        self.shuffle = shuffle

    def batches(self, batch_size):
        """ミニバッチを生成する関数"""
        if self.shuffle:
            idx = np.arange(len(dataset.train_x))
            np.random.shuffle(idx)
            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]
        
        # バッチ取得回数
        n_batches = len(self.train_y) // batch_size
        for ii in range(0, len(self.train_y), batch_size):
            x = self.train_x[ii : ii + batch_size]
            y = self.train_y[ii : ii + batch_size]

            yield self.scaler(x), y
            
class GAN:
    def __init__(self, real_size, z_size, learing_rate, alpha=0.2, beta1=0.5):
        """モデルの定義"""
        # グラフの初期化
        tf.reset_default_graph()

        # 入力データ定義
        self.input_real, self.input_z = model_inputs(real_size, z_size)

        # lossの定義
        self.d_loss, self.g_loss = model_loss(
            self.input_real, self.input_z, real_size[2], alpha=alpha
        )

        # 最適化の定義
        self.d_opt, self.g_opt = model_opt(
            self.d_loss, self.g_loss, learning_rate, beta1
        )
    

# matlab形式: {R, G, B, index}
trainset = loadmat("data/test_32x32.mat")
testset = loadmat("data/train_32x32.mat")

# ハイパーパラメータの設定

real_size = (32, 32, 3)
z_size = 100
learning_rate = 0.0002
batch_size = 128
epochs = 100
alpha = 0.2
beta1 = 0.5

net = GAN(real_size, z_size, learning_rate, alpha=alpha, beta1=beta1)

dataset = Dataset(trainset, testset)

losses, samples = train(net, dataset, epochs, batch_size, figsize=(10, 5))

np.save("losses",  np.array(losses))
np.save("samples",  np.array(samples))