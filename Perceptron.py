import numpy as np
import random
import sys


def input_data(option, input_option_1, input_option_2, input_option_other):
    # use the same method to ingest data for training and testing
    if option == 1:
        file = "train_large.data"
    else:
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
            text = line.replace('\n','').split(',')
            # apply class value according to function parameters and save in array
            if text[4] == input_option_1:
                text[4] = '1'
                array = np.vstack([array, text])
            elif text[4] == input_option_2:
                text[4] = '-1'
                array = np.vstack([array, text])
            elif text[4] == input_option_other:
                continue
    return array


def train(array):
    # initialisation and removal of extra row
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

    # create separate list that contains the shuffled indexes for input file
    r = list(range(len(array)))
    random.shuffle(r)

    # do training for 20 times
    for i in range(0, 20):
        # iterate over all of the lines of the training input data
        for x in range(0, len(array)):
            # update counter for accuracy percentage
            cntTot += 1
            # make prediction with current weights and bias instances
            temp = np.dot(array[r[x]][0:4], weights[0:4]) + b
            # if result of multiplication is negative, it means that prediction was incorrect
            if (temp*array[r[x]][4]) <= 0:
                # update counter for accuracy percentage
                cntA += 1
                # update weights with weight update rule
                weights += np.dot(array[r[x]][0:4], array[r[x]][4])
                # update bias with bias update rule
                b += array[r[x]][4]
            else:
                # update counter for accuracy percentage
                cntB += 1
                continue
    print("Train Correct: ", (cntB/cntTot)*100, "% of Values")
    print("Train Incorrect: ", (cntA/cntTot) * 100, "% of Values")
    return weights,b


def test(weights,b,test_array):
    # initialisations and removal of extra rows
    array = np.delete(test_array, 0, axis=0)
    array = array.astype(np.float)
    res_array = np.array([[0.0, 0.0, 0.0], ])
    cntA = 0
    cntB = 0
    cntTot = 0

    # iterate through whole of test input file data
    for i in range(0,len(array)):
        # update counter for accuracy percentage
        cntTot += 1
        # perform prediction
        temp = np.dot(array[i][0:4], weights[0:4]) + b
        if (temp*array[i][4]) >= 0:
            res_array = np.append(res_array, [[temp, array[i][4], 1]], axis=0)
            # update counter for accuracy percentage
            cntB += 1
        else:
            res_array = np.append(res_array, [[temp, array[i][4], 0]], axis=0)
            # update counter for accuracy percentage
            cntA += 1
    # remove first column of result value test values since it starts with 0s
    res_array = np.delete(res_array, 0, axis=0)
    print("Test Correct Accuracy: ", (cntB/cntTot)*100, "%")
    print("Test Incorrect Accuracy: ", (cntA/cntTot) * 100, "%")
    return res_array


def main():

    # print and receive input from user
    print("1 Class-1 & Class-2")
    print("2 Class-1 & Class-3")
    print("3 Class-2 & Class-3")
    choice = int(input("Enter Binary Prediction Pairs from List: "))

    option1 = "class-1"
    option2 = "class-2"
    option3 = "class-3"

    # take in input data according to input
    if choice == 1:
        array = input_data(1, option1, option2, option3)
    elif choice == 2:
        array = input_data(1, option1, option3, option2)
    elif choice == 3:
        array = input_data(1, option2, option3, option1)
    else:
        print("Choice must be one of the printed options")
        sys.exit()
    # train using the returned input array
    weights, b = train(array)

    # parse test file according to input
    if choice == 1:
        test_array = input_data(0, option1, option2, option3)
    elif choice == 2:
        test_array = input_data(0, option1, option3, option2)
    elif choice == 3:
        test_array = input_data(0, option2, option3, option1)

    # test with computed weights and bias
    res_array = test(weights,b,test_array)
    print("Weights,", weights)
    print("Bias,", b)



if __name__ == "__main__":
    main()