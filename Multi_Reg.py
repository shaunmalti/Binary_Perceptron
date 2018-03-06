import numpy as np
import random
import sys


def input_data(input_option):
    # define target input file
    file = "train_large.data"

    # initialise array
    array = np.array([[0, 0, 0, 0, 0], ])

    # while file is open
    with open(file) as fp:

        # read all lines
        lines = fp.readlines()
        # iterate over each line
        for line in lines:
            # split line contents into array
            text = line.replace('\n', '').split(',')
            # apply class value according to function parameters and save in array
            if text[4] == input_option:
                text[4] = '1'
                array = np.vstack([array, text])
            else:
                text[4] = -1
                array = np.vstack([array, text])
    return array


def train(array, class_num, reg_value):
    # initialise and delete 0-ed row
    array = np.delete(array, (0), axis=0)
    array = array.astype(np.float)
    weights = np.array([0.0, 0.0, 0.0, 0.0])
    b = 0.0

    # create separate list that contains the shuffled indexes for
    # input file to train with random input instances
    r = list(range(len(array)))
    random.shuffle(r)

    # initialise counters for accuracy measure
    cntA = 0
    cntB = 0
    cntTot = 0

    # do training for 20 times
    for i in range(0, 20):
        # iterate over all of the lines of the training input data
        for x in range(0, len(array)):
            # update counter for accuracy percentage
            cntTot += 1
            # make prediction with current weights and bias instances
            temp = np.dot(array[r[x]][0:4], weights[0:4]) + b
            # if result of multiplication is negative, it means that prediction was incorrect
            if (temp * array[r[x]][4]) <= 0:
                # update counter for accuracy percentage
                cntA += 1
                # update weights with weight update rule
                weights = weights + np.dot(array[r[x]][0:4], array[r[x]][4]) - 2*reg_value*(weights)
                # update bias with bias update rule
                b += array[r[x]][4]
            else:
                cntB += 1
                continue
    return weights, b, (cntA/cntTot)


def simple_test(array, weights, b):
    # initialise results array
    temp = [0, 0, 0]
    array = array.astype(np.float)
    # iterate over the 3 weight arrays, performing test and saving result in array
    for i in range(0, len(weights)):
        temp[i] = np.dot(array[0:4], weights[i]) + b[i]
    return temp


def input_test():
    # ingest testing data
    file = "test.data"

    # initialise array
    array = np.array([[0, 0, 0, 0, 0], ])

    # while file is open
    with open(file) as fp:
        # read all lines
        lines = fp.readlines()
        # iterate over each line
        for line in lines:
            # split line contents into array
            text = line.replace('\n', '').split(',')
            # apply class value and save in array
            if text[4] == "class-1":
                text[4] = '1'
                array = np.vstack([array, text])
            elif text[4] == "class-2":
                text[4] = '2'
                array = np.vstack([array, text])
            elif text[4] == "class-3":
                text[4] = '3'
                array = np.vstack([array, text])
    return array


def multi_train(array, weights_1_array, b_1_array, reg_value, class_val):
    # initialisation of training (un)accuracy array
    train_corr = np.array([0],)
    # do training for 100 times
    for i in range(0, 100):
        # train
        weights1, b1, train_inst_cor = (train(array, class_val, reg_value))
        # update corresponding arrays
        weights_1_array = np.vstack([weights_1_array, weights1])
        b_1_array = np.vstack([b_1_array, b1])
        train_corr = np.vstack([train_corr,train_inst_cor])

    # delete first row since it started with a 0-ed row
    weights_1_array = np.delete(weights_1_array, (0), axis=0)
    b_1_array = np.delete(b_1_array, (0), axis=0)
    train_corr = np.delete(train_corr, (0), axis=0)

    # get average of the weights and biases found over the 100 iterations
    final_weights_1 = weights_1_array.mean(axis=0)
    final_b_1 = b_1_array.mean(axis=0)

    # actual average accuracy of training
    final_train_corr = 1-(train_corr.mean())
    print(final_train_corr*100,"% training accuracy for ", class_val)
    return final_weights_1, final_b_1


def main():
    # initialisation for perceptron variables
    weights_1_array = np.array([0, 0, 0, 0], )
    b_1_array = np.array([0], )
    array1 = input_data("class-1")

    # initialisation for perceptron variables
    weights_2_array = np.array([0, 0, 0, 0], )
    b_2_array = np.array([0], )
    array2 = input_data("class-2")

    # initialisation for perceptron variables
    weights_3_array = np.array([0, 0, 0, 0], )
    b_3_array = np.array([0], )
    array3 = input_data("class-3")

    # initialisation of regularisation values
    reg_value = [0.01, 0.1, 1.0, 10.0, 100.0]

    # iterate and perform experiment over all regularisation values
    for wz in range(0,len(reg_value)):
        # train perceptron returning the weights and bias variable
        final_weights_1, final_b_1 = multi_train(array1, weights_1_array, b_1_array, reg_value[wz], class_val="class-1")

        # train perceptron returning the weights and bias variable
        final_weights_2, final_b_2 = multi_train(array2, weights_2_array, b_2_array, reg_value[wz], class_val="class-2")

        # train perceptron returning the weights and bias variable
        final_weights_3, final_b_3 = multi_train(array3, weights_3_array, b_3_array, reg_value[wz], class_val="class-3")

        # store resulting weights and biases in array
        weightsarray = [final_weights_1.tolist(), final_weights_2.tolist(), final_weights_3.tolist()]
        barray = [final_b_1, final_b_2, final_b_3]

        # ingest testing data, this time giving actual class values (1-3)
        testactual = input_test()
        testactual = np.delete(testactual, (0), axis=0)
        testactual = testactual.astype(np.float)

        # initialise counters for accuracy percentage
        tot_1 = 0
        tot_2 = 0
        tot_3 = 0
        pred = [0,0,0]

        # iterate over the length of test data
        for i in range(len(testactual)):
            # perform test on line instance 3 times using all 3 perceptron weights and biases
            temparr = simple_test(testactual[i], weightsarray, barray)

            # get the maximum value from the 3 predictions
            m = temparr.index(max(temparr)) + 1

            # update counters for accuracy percentage
            tot_1 += 1
            tot_2 += 1
            tot_3 += 1

            # iterate over the predictions (3) produced from the test function
            for x in range(0,len(temparr)):
                # true positive
                # if the class value (1-3) is equal to the index (+1) of current value AND
                # if the index of max value found before (m) is equal to the index (+1) of the current value
                if testactual[i][4]==temparr.index(temparr[x])+1 and m == temparr.index(temparr[x])+1:
                    # update corresponding counter for accuracy percentage
                    pred[temparr.index(max(temparr))] += 1
                else:
                    # true negative
                    # the current value is negative AND
                    # the index (+1) of the current value is not equal to the index of the max value (m) AND
                    # the index (+1) of the current value is not equal to the class value (1-3)
                    if temparr[x] < 0 and temparr.index(temparr[x])+1 != m and temparr.index(temparr[x])+1 != testactual[i][4]:
                        # update corresponding counter for accuracy percentage
                        pred[temparr.index(temparr[x])] += 1
                    else:
                        continue
        print("Final Weights for perceptron 1: ", final_weights_1)
        print("Final Weights for perceptron 2: ", final_weights_2)
        print("Final Weights for perceptron 3: ",final_weights_3)
        print((pred[0] / tot_1) * 100, "% class-1 predicted")
        print((pred[1] / tot_2) * 100, "% class-2 predicted")
        print((pred[2] / tot_3) * 100, "% class-3 predicted")

if __name__ == "__main__":
    main()
