import torch
from torch.autograd import Variable
import numpy as np

def get_data():
    train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(train_X).type(dtype),requires_grad=False).view(17,1)
    y = Variable(torch.from_numpy(train_Y).type(dtype),requires_grad=False)
    return X,y

def plot_variable(x,y,z='',**kwargs):
    l = []
    for a in [x,y]:
        if type(a) == Variable:
            l.append(a.data.numpy())
    plt.plot(l[0],l[1],z,**kwargs)

def get_weights():
    w = Variable(torch.randn(1),requires_grad = True)
    b = Variable(torch.randn(1),requires_grad=True)
    return w,b

def simple_network(x, w, b):
    y_pred = torch.matmul(x,w)+b
    return y_pred

def loss_fn(y,y_pred, w, b):
    loss = (y_pred-y).pow(2).sum()
    for param in [w,b]:
        if not param.grad is None: param.grad.data.zero_()
    loss.backward()
    #return loss.data[0]
    return loss


def optimize(learning_rate, w, b):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


def output_to(x, y, pred, w, b, loss, write_wb=False):
    with open("output.csv", 'w') as f:
        f.write("Input, Output, y_pred\n")
        for (xVal, yVal, pred) in zip(x, y, pred):
            f.write(str(xVal.item()) + "," + str(yVal.item()) + "," + str(pred.item()) + '\n')

        if write_wb:
            f.write("\n\nw, b, loss\n")
            for v in [w, b, loss]:
                f.write(str(v.item()) + ",")
        f.write("\n")
        f.close()

learning_rate = 1e-4