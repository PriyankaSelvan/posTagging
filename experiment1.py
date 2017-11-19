import numpy
import time
import os
from collections import Counter
import operator

def count_non_zeros(array):
    count = 0
    for item in array:
        if item != 0:
            count += 1
    print "returning ", count
    return count
def count_non_zeros2(matrix):
    row, col = matrix.shape
    count = 0
    for item1 in range(0, row):
        for item2 in range(0, col):
            if matrix[item1][item2] != 0:
                count += 1
    return count

def main(fold):
    # Files
    filename_train = str(fold)+"train_samples.txt"
    filename_test = str(fold)+"test_samples.txt"

    #Store train statements
    statements_train = []
    tag_count = {}
    baseline_dict = {}
    counter_sentence = 0
    counter_word = 0
    #lineno = -1
    pairs = {}
    with open(filename_train, "r") as f:
        for line in f:
            #lineno += 1
            #print lineno
            if counter_word == 0:
                statements_train.append([])
                statements_train[counter_sentence].append(["_start_", "_START_"])
                if str(statements_train[counter_sentence][counter_word]) in pairs.keys():
                    pairs[str(statements_train[counter_sentence][counter_word])] += 1
                else:
                    pairs[str(statements_train[counter_sentence][counter_word])] = 1
                if "_START_" in tag_count.keys():
                    tag_count["_START_"] += 1
                else:
                    tag_count["_START_"] = 1
                counter_word += 1
                statements_train[counter_sentence].append(line.split()[1:])
                base_word, base_tag = line.split()[1:]
                if base_word in baseline_dict.keys():
                    baseline_dict[base_word].append(base_tag)
                else:
                    baseline_dict[base_word] = [ base_tag ]
                if str(statements_train[counter_sentence][counter_word]) in pairs.keys():
                    pairs[str(statements_train[counter_sentence][counter_word])] += 1
                else:
                    pairs[str(statements_train[counter_sentence][counter_word])] = 1
                #print line
                #print lineno
                if line.split()[2] in tag_count.keys():
                    tag_count[line.split()[2]] += 1
                else:
                    tag_count[line.split()[2]] = 1
                counter_word += 1

            elif line.split():
                statements_train[counter_sentence].append(line.split()[1:])
                base_word, base_tag = line.split()[1:]
                if base_word in baseline_dict.keys():
                    baseline_dict[base_word].append(base_tag)
                else:
                    baseline_dict[base_word] = [base_tag]
                if str(statements_train[counter_sentence][counter_word]) in pairs.keys():
                    pairs[str(statements_train[counter_sentence][counter_word])] += 1
                else:
                    pairs[str(statements_train[counter_sentence][counter_word])] = 1
                if line.split()[2] in tag_count.keys():
                    tag_count[line.split()[2]] += 1
                else:
                    tag_count[line.split()[2]] = 1
                counter_word += 1
            else:
                counter_sentence += 1
                counter_word = 0

    for d in baseline_dict:
        count_dict = Counter(baseline_dict[d])
        baseline_dict[d] = count_dict
    baseline_system_file = open(str(fold)+"baselinesys.txt", "w")

    with open(str(fold)+"test_samples.txt") as f:
        for line in f:
            splits = line.split()
            if splits:
                if splits[1] in baseline_dict:
                    #print max(baseline_dict[splits[1]].iteritems(), key=operator.itemgetter(1))
                    baseline_system_file.write(line.strip() + "\t" + max(baseline_dict[splits[1]].iteritems(), key=operator.itemgetter(1))[0] + "\n")
                else:
                    baseline_system_file.write(line.strip() + "\t" + "NNP" + "\n")
            else:
                baseline_system_file.write("\n")


    #print(statements_train)
    #print(tag_count)
    #print pairs
    print "Done storing training data...."
    number_tags = len(tag_count)
    tags = tag_count.keys()
    tags.sort(reverse=True)
    #print tags
    #print "number of tags ", number_tags
    #calculate tag transition counts
    tt_counts = numpy.zeros(shape=(number_tags, number_tags))
    #print tt_counts
    for i in range(0, len(statements_train)):
        for j in range(0, len(statements_train[i])-1):
            tag1 = statements_train[i][j][1]
            tag2 = statements_train[i][j+1][1]
            tt_counts[tags.index(tag1)][tags.index(tag2)] += 1
    #print tt_counts
    #print tags

    #a = tt_counts[:-1, 1:-1]
    a = tt_counts
    #print a
    '''
    lambda_ = numpy.zeros(number_tags)
    pcont_ = numpy.zeros(number_tags)
    #nkeser smoothing
    for index in range(0, len(tags)):
        if sum(a[index])!= 0:
            print a[index]
            lambda_[index] = count_non_zeros(a[index]) / float(sum(a[index]))
        if sum(a[:][index]) != 0:
            pcont_[index] = count_non_zeros(a[:][index]) / float(count_non_zeros2(a))
    for pari in range(0, len(tags)):
        for parj in range(0, len(tags)):
            den = tag_count[tags[pari]]
            mul = lambda_[pari] * pcont_[parj]
            themax = max(a[pari][parj], 0)
            #print themax
            #print den
            #print mul
            #print "----"
            a[pari][parj] = (themax / float(den)) + (mul)


    print a
    '''
    a = a + 1
    a = a / a.sum(axis=1)[:, None]
    print "Transition probability created...."
    #for i in range(0, number_tags):
    #    row_sum = numpy.sum(a[i])
        #a[i] = numpy.log(a[i]/row_sum)
    #    a[i] = a[i]/float(row_sum)
    #print "beginning a"
    #print a

    #read test sentences
    statements_test = []
    counter_word = 0
    counter_sentence = 0
    #lineno = -1

    with open(filename_test, "r") as f:
        for line in f:
            #lineno += 1
            if counter_word == 0:
                statements_test.append([])
                statements_test[counter_sentence].append("_start_")
                counter_word += 1
                #print lineno
                statements_test[counter_sentence].append(line.split()[1])
                counter_word += 1

            elif line.split():
                statements_test[counter_sentence].append(line.split()[1])
                counter_word += 1
            else:
                counter_sentence += 1
                counter_word = 0
    print "Done storing test sentences..."
    #print statements_test
    # print word_list
    #pairs = []
    #for j in range(0, len(statements_train)):
    #    for k in range(0, len(statements_train[j])):
    #        pairs.append(statements_train[j][k])
    # print pairs

    #word likelihood probabilities for words in test sample
    # followed by viterbi
    sys_file = open(str(fold)+"system-file", "w")
    count = 1

    for i in range(0, len(statements_test)):
        #print "doing sentence ", i
        word_list = [statements_test[i][x] for x in range(0, len(statements_test[i]))]
        likelihood = numpy.zeros(shape=(number_tags, len(word_list)))

        #print likelihood

        for j in range(0, len(word_list)):
            for tag in range(0, number_tags):
                #likelihood[tag][j] = pairs[str([word_list[j], tags[tag]])]
                likelihood[tag][j] = pairs.get(str([word_list[j], tags[tag]]), 0)
            if sum(likelihood[:, j]) == 0:
                likelihood[tags.index("NNP")][j] = 1
                tag_count["NNP"] += 1

        #for j in range(0, len(word_list)):
        #    if sum(likelihood[:, j]) == 0:
                #likelihood[:, j] /= sum(likelihood[:, j])
                #unknown word
        #        likelihood[tags.index("NNP")][j] = 1
        #print likelihood
        for j in range(0, number_tags):
            #sum_row = numpy.sum(likelihood[j])
            #if sum_row == 0:
            #    continue
            #else:
            #    likelihood[j] /= sum_row
            likelihood[j] /= tag_count[tags[j]]





        #print "likelihood sums"

        #print likelihood

        #Viterbi Algorithm

        T = len(word_list)
        K = number_tags
        states = tags
        pi = a[0][:]
        A = a
        B = likelihood

        T1 = numpy.zeros(shape=(K, T))
        T2 = numpy.zeros(shape=(K, T))
        z = numpy.zeros(T)
        X = numpy.zeros(T)

        #print "T = ",T
        #print "K = ", K
        '''
        print "pi is"
        print pi
        print "A is"
        print A
        print "b is"
        print B
        '''

        for ii in range(0, K):
            T1[ii][0] = pi[ii] * B[ii][0]
            T2[ii][0] = 0
        for ii in range(1, T):
            # print "for word ", word_list[ii]
            for jj in range(0, K):
                max_v = -99
                max_k = -1
                for kk in range(0, K):
                    temp = T1[kk][ii-1] * A[kk][jj]
                    #print "temp=", temp
                    if temp > max_v:
                        #print "inside temp>max"
                        max_v = temp
                        max_k = kk
                #print B[jj][ii] * max_v, max_k
                T1[jj][ii] = B[jj][ii] * max_v
                T2[jj][ii] = max_k
        #print "after viterbi"
        #print T1
        #print T2
        max_v = -99
        max_k = -1

        for kk in range(0, K):
            if T1[kk][T-1] > max_v:
                max_v = T1[kk][T-1]
                max_k = kk

        z[T-1] = max_k

        for ii in range(T-1, 0, -1):
            #print "z[", ii-1, "]=t2[", z[ii], "][", ii, "]"
            z[ii-1] = T2[int(z[ii])][ii]
            #print z[ii-1]

        #print "for wordslist ", word_list

        out_list = []

        for tag_id in range(1, len(z)):
            #sys_file.write(tags[int(z[tag_id])]+"\n")
            out_list.append(tags[int(z[tag_id])])
            #print count
            #count += 1


        '''
        print "t1 is"
        print T1
        print "t2 is"
        print T2
        print z
        '''

        #print "for sentence ", i
        for tags_index in range(1, len(statements_test[i])):
            sys_file.write(str(tags_index)+"\t"+str(statements_test[i][tags_index])+"\t"+out_list[tags_index-1]+"\n")
        sys_file.write("\n")

if __name__ == "__main__":
    start_time = time.time()
    #os.system("python makeTestTrain.py 5")
    print "k fold files created..."
    for i in range(1, 6):
        print "fold ", i
        main(i)
        goldf = str(i)+"gold_file"
        systemf = str(i)+"system-file"
        baseline = str(i) + "baselinesys.txt"
        print "-----Baseline accuracy (Tag likelihood model) accuracy-----"
        os.system("python myeval.py "+ goldf+ " "+baseline)
        print "-----HMM based model accuracy-------"
        os.system("python myeval.py "+goldf+" "+systemf)
        #print("--- %s seconds ---" % (time.time() - start_time))
