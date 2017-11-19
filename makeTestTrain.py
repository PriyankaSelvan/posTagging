import random
import sys

k = int(sys.argv[1])

# Store train statements
statements = []

filename = "berp-POS-training.txt"
counter_sentence = 0
counter_word = 0
with open(filename, "r") as f:
    for line in f:
        if counter_word == 0:
            statements.append([])
            statements[counter_sentence].append(line.split()[1:])
            counter_word += 1
            #print line
            #print lineno

        elif line.split():
            statements[counter_sentence].append(line.split()[1:])
            counter_word += 1
        else:
             counter_sentence += 1
             counter_word = 0


#print(statements)
num_sentences = len(statements)


test_ratio = 1/float(k)
train_ratio = 1 - test_ratio

test_num = int(test_ratio * num_sentences)
#print test_num
train_num = num_sentences - test_num
#print range(0, num_sentences)
all_indices = random.sample(range(0, num_sentences), num_sentences)
#print(all_indices)
temp = num_sentences/k
for iti in range(0, k):

    #print(all_indices)
    si = (temp*iti)
    ei = (temp*(iti+1))
    #print si
    #print ei
    test_indices = all_indices[si:ei]
    train_indices = [item for item in all_indices if item not in test_indices]
    #print(len(test_indices))
    #print "----"
    #writing test file
    gold_file = open(str(iti+1)+"gold_file", "w")
    test_file = open(str(iti+1)+"test_samples.txt", "w")
    for sentence_num in test_indices:
        for word_num in range(0, len(statements[sentence_num])):
            test_file.write(str(word_num+1)+"\t"+statements[sentence_num][word_num][0]+"\n")

            gold_file.write(str(word_num+1)+"\t"+statements[sentence_num][word_num][0]+"\t"+statements[sentence_num][word_num][1]+"\n")
        gold_file.write("\n")
        test_file.write("\n")

    #writing train file
    train_file = open(str(iti+1)+"train_samples.txt", "w")
    for sentence_num in train_indices:
        for word_num in range(0, len(statements[sentence_num])):
            train_file.write(str(word_num+1)+"\t"+statements[sentence_num][word_num][0]+"\t"+statements[sentence_num][word_num][1]+"\n")
        train_file.write("\n")
