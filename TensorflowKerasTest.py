
import tensorflow as tf
import pandas as pd
import numpy as np
from sktime.transformations.series.vmd import VmdTransformer
from sklearn.model_selection import train_test_split
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Model(tf.Module):
    def __init__(self):
        rand_init = tf.random.uniform(shape=[3], minval=0, maxval=5, seed=22)
        self.w_q = tf.Variable(rand_init[0])
        self.w_l = tf.Variable(rand_init[1])
        self.b = tf.Variable(rand_init[2])
    
    @tf.function
    def __call__(self, x):
        return self.w_q * (x**2) + self.w_l * x + self.b

def plot_preds(x, y, f, model, title):
    plt.figure()
    plt.plot(x,y,'.', label= 'data')
    plt.plot(x,f(x), label = 'ground truth')
    plt.plot (x, model(x), label = 'predictions')
    plt.title(title)
    plt.legend()
    plt.show()

def f(x):
    y = x**2 +2*x -5
    return y

def mse_loss (y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))

def main ():
    matplotlib.rcParams['figure.figsize'] = [9,6]
    x = tf.linspace(-2,2,201)
    x= tf.cast (x,tf.float32)

    y = f(x) + tf.random.normal(shape=[201])
    plt.plot(x.numpy(), y.numpy(), '.', label="Data")
    plt.plot(x,f(x), label = 'Ground Truth')
    plt.legend();
    plt.show()
    
    quad_model = Model()
    plot_preds(x, y, f, quad_model, "before training")

    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.shuffle(buffer_size=x.shape[0]).batch(batch_size)

    epochs = 100
    learning_rate = 0.01
    losses = []

    for epoch in range(epochs):
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                batch_loss = mse_loss(quad_model(x_batch), y_batch)
            grads = tape.gradient(batch_loss, quad_model.variables)

            for g,v in zip(grads,quad_model.variables):
                v.assign_sub(learning_rate*g)
        loss = mse_loss(quad_model(x), y)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f'Mean squared error for step {epochs}: {loss.numpy():0.3f}')    

    print("\n")
    plt.plot(range(epochs), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title('MSE loss vs training iterations');
    plt.show()
    plot_preds(x, y, f, quad_model, 'After training')
    
    new_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.stack([x,x**2], axis =1)),
        tf.keras.layers.Dense(units=1, kernel_initializer= tf.random.normal)
    ])

    new_model.compile(
        loss = tf.keras.losses.MSE,
        optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
    )
    history = new_model.fit(x,y, epochs=100, batch_size=32,verbose=0)
    new_model.save('./my_new_model.keras')
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylim([0, max(plt.ylim())])
    plt.ylabel('Loss [Mean Squared Error]')
    plt.title('Keras training progress');
    plt.show()

    plot_preds(x, y, f, new_model, 'After Training: Keras')

if __name__ == "__main__" :
    main()