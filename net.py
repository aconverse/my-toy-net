import scipy.io
import numpy as np
import random


#def relu(x):
#    return np.maximum(0, x)
#
#def step(x):
#    return x >= 0

def sig(x):
    return 1 / (1 + np.exp(-x))

def dsig(x):
    # todo dsig of sig? rely on previosu computation of sig(x) which is act
    s = sig(x)
    return s * (1 - s)

def dsig_sig(s):
    return s * (1 - s)

def feedforward(img, weights, biases):
    activations = [img / 256]
    for w, b in zip(weights, biases):
        activations.append(sig(activations[-1] @ w + b))
    return np.array(activations, dtype=object)

def classify(img, weights, biases):
    # same as feed forward but without history
    activations = img / 256
    for w, b in zip(weights, biases):
        activations = sig(activations @ w + b)
    return activations

def score_batch(dataset, weights, biases):
    total_correct = 0
    total_cost = 0.
    for img, label in dataset:
        act = classify(img, weights, biases)
        selected = np.argmax(act)
        total_correct += 1 if selected == label else 0
        total_cost += sse(act, one_at(label, 10))
    return (total_correct, total_cost)

def backprop(weights, biases, act, desired):
    # C = sum(a[j] - y[j])
    # a[j] = activation_func( z[j] )
    # z[j] = sum ( w[k][j] * a[L-1][k] ) + b[j]

    assert(desired.sum() == 1)
    assert(len(weights) == len(biases))
    assert(len(weights) == len(act)-1)

    # dout is the dC/da[L]
    dout = 2 * (act[-1] - desired)

    del_w = np.zeros(weights.shape, dtype=object)
    del_b = np.zeros(biases.shape, dtype=object)

    for L in range(len(act) - 2, -1, -1):
        dout_times_da_dz = dout * dsig_sig(act[L+1])
        del_b[L] = dout_times_da_dz 
        del_w[L] = np.outer(act[L], dout_times_da_dz)
        dout = weights[L] @ dout_times_da_dz
    return np.array([del_w, del_b], dtype=object)

def apply_minibatch(weights, biases, batch, learning_factor):
    cum_upd = np.array([np.array([np.zeros(w.shape) for w in weights], dtype=object), np.array([np.zeros(b.shape) for b in biases], dtype=object)])
    for (img, label) in batch:
            act = feedforward(img, weights, biases)
            upd = backprop(weights, biases, act, one_at(label, 10))
            cum_upd += upd
    cum_upd *= -learning_factor / len(batch)
    weights += cum_upd[0]
    biases += cum_upd[1]

def one_at(n, size):
    rv = np.zeros(size)
    rv[n] = 1
    return rv

def sse(a, b):
    tmp = (a - b)
    return np.dot(tmp, tmp)

def show(img):
    d = int(np.sqrt(img.size))
    twod = img.reshape((d, d)).transpose()
    for row in twod:
        print("".join(['#' if px > 128 else '.' for px in row]))

def train(weights, biases, opts, training_data, holdback_data):
    thresh = opts['thresh']
    learning_factor = opts['learning_factor']
    mini_batch_size = opts['mini_batch_size']
    trained_since_last_score = 0
    # this batch structure is inspired a bit by the neuralnetworksanddeeplearning.com book
    for batch in range(0, opts['max_full_batches']):
        random.shuffle(training_data)
        for k in range(0, len(training_data), mini_batch_size):
            mini_batch = training_data[k:k+mini_batch_size]
            apply_minibatch(weights, biases, mini_batch, learning_factor)
            trained_since_last_score += len(mini_batch)
            if trained_since_last_score >= len(holdback_data):
                total_correct, total_cost = score_batch(holdback_data, weights, biases)
                print("score correct {} of {} cost {}".format(total_correct, len(holdback_data), total_cost))
                if total_correct >= thresh * len(holdback_data):
                    return
                trained_since_last_score = 0

def learn_and_score(training_data, test_data, opts):
    holdback_size = opts['holdback_size']
    training_data, holdback_data = training_data[:-holdback_size], training_data[-holdback_size:]

    dims = opts['dims']
    weights = np.array([np.random.randn(d[0], d[1]) for d in dims], dtype=object)
    biases = np.array([np.random.randn(d[1]) for d in dims], dtype=object)
    train(weights, biases, opts, training_data, holdback_data)
    total_correct, total_cost = score_batch(test_data, weights, biases)
    print("score correct {} of {} cost {}".format(total_correct, len(test_data), total_cost))

def load_data():
    # unpack matlab zip from https://greg-cohen.com/datasets/emnist/ here 
    matfile = scipy.io.loadmat("matlab/emnist-digits.mat")['dataset'][0,0]

    trainDS = matfile[0][0,0]
    testDS = matfile[1][0,0]

    training_data = list(zip(trainDS[0], trainDS[1][:,0]))
    test_data = list(zip(testDS[0], testDS[1][:,0]))
    return training_data, test_data

training_data, test_data = load_data()
opts = {
    'dims': [(28*28, 16), (16, 16), (16, 10)],
    'holdback_size': min(len(training_data) // 10, 10000),
    'thresh': 0.95,
    'learning_factor': 0.5,
    'mini_batch_size': 10,
    'max_full_batches': 10,
}

# np.set_printoptions(precision=4)
learn_and_score(training_data, test_data, opts)
