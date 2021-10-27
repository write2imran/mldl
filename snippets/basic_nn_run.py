from functions import basic_nn
from functions.basic_nn import *


def execute_nn():
    x, y = get_data()  # x - represents training data,y - represents target variables
    w, b = get_weights()  # w,b - Learnable parameters

    # Wrote X, Y Values to File

    for i in range(500):
        y_pred = simple_network(x, w, b)  # function which computes wx + b
        loss = loss_fn(y, y_pred, w, b)  # calculates sum of the squared differences of y and y_pred

        output_to(x, y, y_pred, w, b, loss, True)   # write output to file

        #if i % 50 == 0:
        #    print(loss)

        optimize(learning_rate, w, b)  # Adjust w,b to minimize the loss


if __name__ == '__main__':
    execute_nn()
    print("-----------done-----------")
