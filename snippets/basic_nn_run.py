import torch

from functions import basic_nn
from functions.basic_nn import *
import datetime

def execute_nn():
    now = datetime.datetime.now()
    dt = now.strftime("%Y-%m-%d-%H:%M:%S")

    default_clean = True
    if default_clean:
        try:
            os.remove("output.csv")
            os.remove("parameters.csv")
        except FileNotFoundError:
            print("One of th file doesn't exist")

    x, y = get_data()  # x - represents training data,y - represents target variables
    #w, b = get_weights()  # w,b - Learnable parameters

    w = torch.tensor([1.13093483448028],  requires_grad=True)  # initialize with fixed w for debugging
    b = torch.tensor([0.851598381996154],  requires_grad=True)  # initialize with fixed w for debugging

    # Wrote X, Y Values to File
    for i in range(500):
        y_pred = simple_network(x, w, b)  # function which computes wx + b
        loss = loss_fn(y, y_pred, w, b)  # calculates sum of the squared differences of y and y_pred

        output_io(x, y, y_pred)   # write output to file
        output_wb_loss("CASE-"+str(i)+"-"+dt, w, b, loss)

        #if i % 50 == 0:
        #    print(loss)

        optimize(learning_rate, w, b)  # Adjust w,b to minimize the loss


if __name__ == '__main__':
    execute_nn()
    print("-----------done-----------")
