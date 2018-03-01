import numpy as np
import random
import sys


def input_data(input_option):
    file = "train_large.data"
    array = np.array([[0, 0, 0, 0, 0], ])
    with open(file) as fp:
        lines = fp.readlines()
        for line in lines:
            text = line.replace('\n', '').split(',')
            if text[4] == input_option:
                text[4] = '1'
                array = np.vstack([array, text])
            # elif text[4] == "class-2":
            #     text[4] = '2'
            #     array = np.vstack([array, text])
            # elif text[4] == "class-3":
            #     text[4] = '3'
            #     array = np.vstack([array, text])
            else:
                text[4] = -1
                array = np.vstack([array, text])
    return array


def input_test_new():
    file = "test.data"
    array = np.array([[0, 0, 0, 0, 0], ])
    with open(file) as fp:
        lines = fp.readlines()
        for line in lines:
            text = line.replace('\n', '').split(',')
            if text[4] == "class-1":
                text[4] = '1'
                array = np.vstack([array, text])
            elif text[4] == "class-2":
                text[4] = '2'
                array = np.vstack([array, text])
            elif text[4] == "class-3":
                text[4] = '3'
                array = np.vstack([array, text])
            else:
                text[4] = -1
                array = np.vstack([array, text])
    return array


def round_sqr_weights(wt):
    for i in range(0,len(wt)):
        wt[i] **= 2
        wt[i] = np.round(wt[i],2)
    return wt


def train(array, class_num, reg_value):
    array = np.delete(array, (0), axis=0)  # remove first column of input values
    array = array.astype(np.float)
    weights = np.array([0.0, 0.0, 0.0, 0.0])  # 0-ed initial weights
    b = 0.0  # bias variable
    cntA = 0
    cntB = 0
    cntTot = 0
    for i in range(0, 20):
        r = list(range(len(array)))
        random.shuffle(r)
        for x in range(0, len(array)):
            cntTot += 1
            temp = np.dot(array[r[x]][0:4], weights[0:4]) + b
            if (temp * array[r[x]][4]) <= 0:

                cntA += 1
                # print(2*reg_value*(np.sum(np.around(weights**2,decimals=2))))
                weights = np.around(weights + np.dot(array[r[x]][0:4], array[r[x]][4]) - 2*reg_value*(weights),2) # np.around(weights**2,decimals=2
                # weights = np.around(weights,decimals=2)
                b += array[r[x]][4]

            else:
                cntB += 1
                continue
    # print("Train Correct: ", (cntB / cntTot) * 100, "% of Values ", class_num, "Iteration: ", iteration_num)
    # print("Train Incorrect: ", (cntA / cntTot) * 100, "% of Values", class_num, "Iteration: ", iteration_num)
    cntA = 0
    cntB = 0
    cntTot = 0
    return weights, b


def simple_test(array, weights, b):
    temp = [0, 0, 0]
    array = array.astype(np.float)
    for i in range(0, len(weights)):
        temp[i] = np.dot(array[0:4], weights[i]) + b[i]
    return temp


def input_test():
    file = "test.data"
    array = np.array([[0, 0, 0, 0, 0], ])
    with open(file) as fp:
        lines = fp.readlines()
        for line in lines:
            text = line.replace('\n', '').split(',')
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
    weights1, b1 = (train(array, class_val, reg_value))
    weights_1_array = np.vstack([weights_1_array, weights1])
    b_1_array = np.vstack([b_1_array, b1])
    weights_1_array = np.delete(weights_1_array, (0), axis=0)

    b_1_array = np.delete(b_1_array, (0), axis=0)
    final_weights_1 = weights_1_array.mean(axis=0)
    final_b_1 = b_1_array.mean(axis=0)
    return final_weights_1, final_b_1


def main():
    weights_1_array = np.array([0, 0, 0, 0], )
    b_1_array = np.array([0], )
    array1 = input_data("class-1")

    weights_2_array = np.array([0, 0, 0, 0], )
    b_2_array = np.array([0], )
    array2 = input_data("class-2")

    weights_3_array = np.array([0, 0, 0, 0], )
    b_3_array = np.array([0], )
    array3 = input_data("class-3")
    reg_value = [0.01, 0.1, 1.0, 10.0, 100.0]

    weights_1_reg_val = np.array([0, 0, 0, 0], )
    weights_2_reg_val = np.array([0, 0, 0, 0], )
    weights_3_reg_val = np.array([0, 0, 0, 0], )

    # for w in range(0,len(reg_value)):
    final_weights_1, final_b_1 = multi_train(array1, weights_1_array, b_1_array, reg_value[0], class_val="class-1")
    weights_1_reg_val = np.vstack([weights_1_reg_val, final_weights_1])

    final_weights_2, final_b_2 = multi_train(array2, weights_2_array, b_2_array, reg_value[0], class_val="class-2")
    weights_2_reg_val = np.vstack([weights_2_reg_val, final_weights_2])
        #
    final_weights_3, final_b_3 = multi_train(array3, weights_3_array, b_3_array, reg_value[0], class_val="class-3")
    weights_3_reg_val = np.vstack([weights_3_reg_val, final_weights_3])

    weights_1_reg_val = np.delete(weights_1_reg_val, (0), axis=0)
    weights_2_reg_val = np.delete(weights_2_reg_val, (0), axis=0)
    weights_3_reg_val = np.delete(weights_3_reg_val, (0), axis=0)

    weightsarray = [final_weights_1.tolist(), final_weights_2.tolist(), final_weights_3.tolist()]
    barray = [final_b_1, final_b_2, final_b_3]

    testactual = input_test_new()
    testactual = np.delete(testactual, (0), axis=0)
    testactual = testactual.astype(np.float)
    #
    tot_1 = 0
    tot_2 = 0
    tot_3 = 0
    pred_1 = 0
    pred_2 = 0
    pred_3 = 0
    #
    for i in range(len(testactual)):
        temparr = simple_test(testactual[i], weightsarray, barray)
        m = temparr.index(max(temparr)) + 1
        if (testactual[i][4] == 1):
            tot_1 += 1
        elif (testactual[i][4] == 2):
            tot_2 += 1
        elif (testactual[i][4] == 3):
            tot_3 += 1
        if (int(testactual[i][4]) == m):
            # print(m, " CORRECT")
            if (testactual[i][4] == 1):
                pred_1 += 1
            elif (testactual[i][4] == 2):
                pred_2 += 1
            elif (testactual[i][4] == 3):
                pred_3 += 1
        else:
            continue
            # print("INC")
        temparr = [0, 0, 0]
    print("Final Weights for perceptron 1: ", final_weights_1)
    print("Final Weights for perceptron 2: ", final_weights_2)
    print("Final Weights for perceptron 3: ",final_weights_3)
    #
    print((pred_1 / tot_1) * 100, "% class-1 predicted")
    print((pred_2 / tot_2) * 100, "% class-2 predicted")
    print((pred_3 / tot_3) * 100, "% class-3 predicted")


#     weights += np.dot(array[r[x]][0:4], array[r[x]][4]) + 2*reg_value*(np.sum(weights**2))

if __name__ == "__main__":
    main()
