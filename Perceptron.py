import numpy as np
import random
import sys


def input_data(option, input_option_1, input_option_2, input_option_other):
    if option == 1:
        file = "train_large.data"
    else:
        file = "test.data"
    array = np.array([[0, 0, 0, 0, 0], ])
    with open(file) as fp:
        lines = fp.readlines()
        for line in lines:
            text = line.replace('\n','').split(',')
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
    array = np.delete(array, (0), axis=0)  # remove first column of input values
    array = array.astype(np.float)
    # weights = np.random.rand(4)  # randomised initial weights
    weights = np.array([0.0, 0.0, 0.0, 0.0])
    b = 0.0  # bias variable
    r = list(range(len(array)))  # TODO CHANGE BACK TO INCLUDE RANDOM INPUT
    random.shuffle(r)
    cntA = 0
    cntB = 0
    cntTot = 0
    for i in range(0, 20):
        for x in range(0, len(array)):
            cntTot += 1
            temp = np.dot(array[r[x]][0:4], weights[0:4]) + b # temp = np.dot(array[r[x]][0:4], weights[0:4]) + b
            if (temp*array[r[x]][4]) <= 0:
                cntA += 1 # wrong result
                weights += np.dot(array[r[x]][0:4], array[r[x]][4])
                b += array[r[x]][4]
            else:
                cntB += 1
                continue
    print("Train Correct: ", (cntB/cntTot)*100, "% of Values")
    print("Train Incorrect: ", (cntA/cntTot) * 100, "% of Values")
    return weights,b


def test(weights,b,test_array):
    array = np.delete(test_array, 0, axis=0)  # remove first column of input test values
    array = array.astype(np.float)
    res_array = np.array([[0.0, 0.0, 0.0], ])  # 0-ed initial weights
    cntA = 0
    cntB = 0
    cntTot = 0
    for i in range(0,len(array)):
        cntTot += 1
        temp = np.dot(array[i][0:4], weights[0:4]) + b
        if (temp*array[i][4]) >= 0:  # if sign produced is positive then prediction is correct
            res_array = np.append(res_array, [[temp, array[i][4], 1]], axis=0)
            # print(str(temp) + "," + str(array[i][4]) + "," + str(1))
            cntB += 1
        else: # i.e. when is incorrect
            res_array = np.append(res_array, [[temp, array[i][4], 0]], axis=0)
            # print(str(temp) + "," + str(array[i][4]) + "," + str(0))
            cntA += 1
    res_array = np.delete(res_array, 0, axis=0)  # remove first column of result value test values
    print("Test Correct Accuracy: ", (cntB/cntTot)*100, "%")
    print("Test Incorrect Accuracy: ", (cntA/cntTot) * 100, "%")
    return res_array


def main():
    print("1 Class-1 & Class-2")
    print("2 Class-1 & Class-3")
    print("3 Class-2 & Class-3")
    choice = int(input("Enter Binary Prediction Pairs from List: "))

    option1 = "class-1"
    option2 = "class-2"
    option3 = "class-3"

    if choice == 1:
        array = input_data(1, option1, option2, option3)
    elif choice == 2:
        array = input_data(1, option1, option3, option2)
    elif choice == 3:
        array = input_data(1, option2, option3, option1)
    else:
        print("Choice must be one of the printed options")
        sys.exit()
    weights, b = train(array)

    if choice == 1:
        test_array = input_data(0, option1, option2, option3)
    elif choice == 2:
        test_array = input_data(0, option1, option3, option2)
    elif choice == 3:
        test_array = input_data(0, option2, option3, option1)
    res_array = test(weights,b,test_array)
    print("Weights,", weights)
    print("Bias,", b)



if __name__ == "__main__":
    main()