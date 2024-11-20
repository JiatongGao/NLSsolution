# tensorflow 2.x, python 3.11

import sys

sys.path.insert(0, '../../Utilities/')

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow can access the GPU!")
else:
    print("TensorFlow cannot find any GPUs.")
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.random.set_seed(1234)


class PhysicsInformedNN:
    def __init__(self, x0, y0, lb, ub, tb, XY_f, layers, u0, v0):
        # initial condition
        X0 = np.concatenate([x0, np.zeros_like(x0)], axis=1)
        Y0 = np.concatenate([y0, np.zeros_like(y0)], axis=1)
        # boundary condition
        X_lb = np.concatenate([lb[0] * np.ones_like(tb), tb], axis=1)
        X_ub = np.concatenate([ub[0] * np.ones_like(tb), tb], axis=1)
        Y_lb = np.concatenate([lb[1] * np.ones_like(tb), tb], axis=1)
        Y_ub = np.concatenate([ub[1] * np.ones_like(tb), tb], axis=1)


        self.lb = lb
        self.ub = ub


        # x0, y0 and t0
        self.x0 = X0[:, 0:1]
        self.y0 = Y0[:, 0:1]
        self.t0 = X0[:, 1:2]
        # xlb ylb and tb
        self.x_lb = X_lb[:, 0:1]
        self.y_lb = Y_lb[:, 0:1]
        self.t_lb = X_lb[:, 1:2]

        self.x_ub = X_ub[:, 0:1]
        self.y_ub = Y_ub[:, 0:1]
        self.t_ub = X_ub[:, 1:2]


        self.xy_f = XY_f
        self.x_f = XY_f[:, 0:1]
        self.y_f = XY_f[:, 1:2]
        self.t_f = XY_f[:, 2:3]


        self.u0 = u0
        self.v0 = v0

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

        self.trainable_variables = []
        for W, b in zip(self.weights, self.biases):
            self.trainable_variables.append(W)
            self.trainable_variables.append(b)

    def loss(self, x0, y0, t0, u0, v0, x_lb, y_lb, t_lb, x_ub, y_ub, t_ub, xy_f, x_f, y_f, t_f):
        u0_pred, v0_pred, _, _, _, _ = self.net_uv(x0, y0, t0)
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred, u_y_lb_pred, v_y_lb_pred = self.net_uv(x_lb, y_lb, t_lb)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred, u_y_ub_pred, v_y_ub_pred = self.net_uv(x_ub, y_ub, t_ub)
        f_u_pred, f_v_pred = self.net_f_uv(x_f, y_f, t_f)

        loss_value = tf.reduce_mean(tf.square(u0 - u0_pred)) + \
                     tf.reduce_mean(tf.square(v0 - v0_pred)) + \
                     tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
                     tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
                     tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + \
                     tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred)) + \
                     tf.reduce_mean(tf.square(u_y_lb_pred - u_y_ub_pred)) + \
                     tf.reduce_mean(tf.square(v_y_lb_pred - v_y_ub_pred)) + \
                     tf.reduce_mean(tf.square(f_u_pred)) + \
                     tf.reduce_mean(tf.square(f_v_pred))

        return loss_value



    # function define weights and biases
    def initialize_NN(self,layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        weight = tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev))
        return weight

    def neural_net(self, X, weights, biases):
        num_layers = len(weights)+1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Z = tf.add(tf.matmul(H, W), b)
        return Z

    def net_uv(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            X = tf.concat([x, y, t], axis=1)
            uv = self.neural_net(X, self.weights, self.biases)
            u = uv[:,0:1]
            v = uv[:,1:2]
        # compute derivative
        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        del tape
        return u, v, u_x, v_x, u_y, v_y

    def net_f_uv(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            u, v, u_x, v_x, u_y, v_y = self.net_uv(x, y, t)
            # calculate the derivative of t
            u_t = tape.gradient(u, t)
            v_t = tape.gradient(v, t)
        # calculate the second derivative
        u_xx = tape.gradient(u_x, x)
        v_xx = tape.gradient(v_x, x)
        u_yy = tape.gradient(u_y, y)
        v_yy = tape.gradient(v_y, y)

        f_u = u_t + v_xx + v_yy + v * (v ** 2 - 3 * u**2)
        f_v = v_t - u_xx - u_yy + u * (u ** 2 - 3 * v**2)

        del tape
        return f_u, f_v

    def train_step(self, x0, y0, t0, u0, v0, x_lb, y_lb, t_lb, x_ub, y_ub, t_ub, xy_f, x_f, y_f, t_f):
        x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
        y0 = tf.convert_to_tensor(y0, dtype=tf.float32)
        t0 = tf.convert_to_tensor(t0, dtype=tf.float32)
        u0 = tf.convert_to_tensor(u0, dtype=tf.float32)
        v0 = tf.convert_to_tensor(v0, dtype=tf.float32)
        x_lb = tf.convert_to_tensor(x_lb, dtype=tf.float32)
        y_lb = tf.convert_to_tensor(y_lb, dtype=tf.float32)
        t_lb = tf.convert_to_tensor(t_lb, dtype=tf.float32)
        x_ub = tf.convert_to_tensor(x_ub, dtype=tf.float32)
        y_ub = tf.convert_to_tensor(y_ub, dtype=tf.float32)
        t_ub = tf.convert_to_tensor(t_ub, dtype=tf.float32)
        xy_f = tf.convert_to_tensor(xy_f, dtype=tf.float32)
        x_f = tf.convert_to_tensor(x_f, dtype=tf.float32)
        y_f = tf.convert_to_tensor(y_f, dtype=tf.float32)
        t_f = tf.convert_to_tensor(t_f, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # 调整loss函数的调用，以包括y维度
            loss_value = self.loss(x0, y0, t0, u0, v0, x_lb, y_lb, t_lb, x_ub, y_ub, t_ub, xy_f, x_f, y_f, t_f)
        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss_value

    def train(self,nIter):
        start_time = time.time()
        self.loss_history = []
        for it in range(nIter):
            loss = self.train_step(self.x0, self.y0, self.t0, self.u0, self.v0,
                                   self.x_lb, self.y_lb, self.t_lb, self.x_ub, self.y_ub,
                                   self.t_ub, self.xy_f, self.x_f, self.y_f, self.t_f)
            self.loss_history.append(loss.numpy())

            if it % 10 == 0:
                elapsed = time.time() - start_time
                print(f'It: {it}, Loss: {loss.numpy():.3e}, Time: {elapsed:.2f}')
                start_time = time.time()

    def predict(self, XY_star):
        x_star = tf.convert_to_tensor(XY_star[:, 0:1], dtype=tf.float32)
        y_star = tf.convert_to_tensor(XY_star[:, 1:2], dtype=tf.float32)
        t_star = tf.convert_to_tensor(XY_star[:, 2:3], dtype=tf.float32)

        u_star, v_star, _, _, _, _ = self.net_uv(x_star, y_star, t_star)
        f_u_star, f_v_star = self.net_f_uv(x_star, y_star, t_star)

        return u_star, v_star, f_u_star, f_v_star

if __name__ == "__main__":
    noise = 0.0

    # domain boundary

    lb = np.array([-5.0, -3.0, 0.0])  #(x,y,t)
    ub = np.array([5.0, 3.0, np.pi / 2])

    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [3, 100, 100, 100, 100, 3]

    data = scipy.io.loadmat('../Data/datanls.mat')
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    y = data['y'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    X, T = np.meshgrid(x, t)
    Y, _ = np.meshgrid(y, t)

    XY_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))


    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    idx_y = np.random.choice(y.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    y0 = y[idx_y, :]
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]

    XY_f = lb + (ub - lb) * lhs(3, N_f)



    model = PhysicsInformedNN(x0, y0, lb, ub, tb, XY_f, layers, u0, v0)

    start_time = time.time()
    model.train(20000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    ################
    # loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(model.loss_history, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()

    savefig('./figuresby2.0.1/loss1')

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(XY_star)
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)


    # contour plot

    target_ts = [0.58, 0.75, 0.98, 1.57]
    fig, axs = plt.subplots(1, len(target_ts), figsize=(20, 5))
    for i, time in enumerate(target_ts):
        x_lin = np.linspace(x.min(), x.max(), 100)
        y_lin = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(x_lin, y_lin)
        # 准备模型输入，注意每个点都包含相同的 t_fixed 值
        X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], np.full_like(X.flatten()[:, None], time)))
        # 使用模型进行预测
        u_pred, v_pred, _, _ = model.predict(X_star)  # 确保model.predict能够处理这种输入格式
        h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
        # 插值预测结果，这里我们只以 u_pred 为例
        H_pred = griddata((X_star[:, 0], X_star[:, 1]), h_pred.flatten(), (X, Y), method='cubic')
        # Plot
        cf = axs[i].contourf(X, Y, H_pred.reshape(X.shape), levels=100, cmap='jet')
        fig.colorbar(cf, ax=axs[i])
        axs[i].set_title(f'Predicted H norm at t={time}')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')

    plt.tight_layout()
    savefig('./figuresby2.0.1/hnorms')



































