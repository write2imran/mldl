from functions import basic_nn
from functions.basic_nn import *


def execute_nn():
    x, y = get_data()  # x - represents training data,y - represents target variables
    w, b = get_weights()  # w,b - Learnable parameters

    # Wrote X, Y Values to File

    with open("output.txt", 'w') as f:
        f.write("Input, Output\n")
        for (xVal, yVal) in zip(x, y):
            f.write(str(xVal.item()) + "," + str(yVal.item()) + '\n')

        f.write("\n\nInitial w and b\n")
        for v in [w, b]:
            f.write(str(v.item()) + "\n")
        f.write("\n")
        f.close()

    for i in range(500):
        y_pred = simple_network(x, w, b)  # function which computes wx + b
        loss = loss_fn(y, y_pred, w, b)  # calculates sum of the squared differences of y and y_pred
        if i % 50 == 0:
            print(loss)
        optimize(learning_rate)  # Adjust w,b to minimize the loss


if __name__ == '__main__':
    execute_nn()
    print("-----------done-----------")
