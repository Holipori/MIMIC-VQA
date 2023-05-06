import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np

import os


def preprocess_dataset():
    os.mkdir('temp')
    path = 'temp/mimic_pair_questions_temp.csv'
    # read csv file using pandas
    df = pd.read_csv(path)
    mimic_all_path = '../mimic_all.csv'
    d_all = pd.read_csv(mimic_all_path)

    wordset = set()
    answer_count = {}
    # obtain labelse first
    for i in tqdm(range(len(df))):
        question = df.iloc[i]['question'].replace('?',' ?')
        answers = df.iloc[i]['answer'].replace('.','')
        answers = answers.split(', ')
        for answer in answers:
            if answer not in answer_count:
                answer_count[answer] = 1
            else:
                answer_count[answer] += 1
        wordset.update(question.split())
        wordset.update(answers)
    wordset = list(wordset)

    # remove answers that count is less than 5 from ans2label
    label2ans = [label for label in answer_count if answer_count[label] >= 5]
    wordset.sort()
    label2ans.sort()
    # get word2id
    word2id = {word: i for i, word in enumerate(wordset)}
    # transform labelset to dict
    ans2label = {label: i for i, label in enumerate(label2ans)}
    answerset = set(label2ans)


    total_dataset = []
    for i in tqdm(range(len(df))):
        record = {}
        question = df.iloc[i]['question']
        answer = df.iloc[i]['answer'].replace('.', '')
        answer = answer.split(', ')
        for ans in answer:
            if ans not in answerset:
                answer.remove(ans)
        if answer == []:
            continue
        subject_id = df.iloc[i]['subject_id']
        study_id = df.iloc[i]['study_id']
        dicom_id = d_all[d_all['study_id'] == study_id]
        dicom_id = dicom_id[dicom_id['view'].isin(['postero-anterior','antero-posterior'])]['dicom_id'].values[0]
        labels = [ans2label[ans] for ans in answer]
        # set scores to all 1.0
        scores = [1.0] * len(labels)

        record['subject_id'] = subject_id
        record['study_id'] = study_id
        record['dicom_id'] = dicom_id
        record['question'] = question
        record['question_type'] = df.iloc[i]['question_type']
        record['answer'] = {'labels': labels, 'scores': scores, 'answer': answer}
        total_dataset.append(record)
    dictionary = [word2id, wordset]
    # split the datasets
    train_dataset = total_dataset[:int(len(total_dataset) * 0.8)]
    val_dataset = total_dataset[int(len(total_dataset) * 0.8):int(len(total_dataset) * 0.9)]
    test_dataset = total_dataset[int(len(total_dataset) * 0.9):]
    print('train:', len(train_dataset))
    print('val:', len(val_dataset))
    print('test:', len(test_dataset))
    # save to pickle file
    train_path = 'temp/mimic_dataset_train.pkl'
    val_path = 'temp/mimic_dataset_val.pkl'
    test_path = 'temp/mimic_dataset_test.pkl'
    with open(train_path, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val_dataset, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_dataset, f)
    #save dictionary
    dictionary_path = 'temp/mimic_dictionary.pkl'
    with open(dictionary_path, 'wb') as f:
        pickle.dump(dictionary, f)
    # save label2ans
    label2ans_path = 'temp/mimic_label2ans.pkl'
    with open(label2ans_path, 'wb') as f:
        pickle.dump(label2ans, f)
    # save ans2label
    ans2label_path = 'temp/mimic_ans2label.pkl'
    with open(ans2label_path, 'wb') as f:
        pickle.dump(ans2label, f)

    # save label_count
    label_count_path = 'temp/mimic_label_count.pkl'
    with open(label_count_path, 'wb') as f:
        pickle.dump(answer_count, f)


def remove_low_freq_labels():
    train_path = 'temp/mimic_dataset_train.pkl'
    val_path = 'temp/mimic_dataset_val.pkl'
    test_path = 'temp/mimic_dataset_test.pkl'
    label2ans_path = 'temp/mimic_label2ans.pkl'
    ans2label_path = 'temp/mimic_ans2label.pkl'
    label_count_path = 'temp/mimic_label_count.pkl'

    train = pickle.load(open(train_path, 'rb'))
    val = pickle.load(open(val_path, 'rb'))
    test = pickle.load(open(test_path, 'rb'))
    label2ans = pickle.load(open(label2ans_path, 'rb'))
    ans2label = pickle.load(open(ans2label_path, 'rb'))
    label_count = pickle.load(open(label_count_path, 'rb'))

    labels_need_to_remove_total = []
    for split in [train, val, test]:
        labels_need_to_remove = label2ans.copy() + label2ans.copy()
        for i in range(len(split)):
            for ans in split[i]['answer']['answer']:
                try:
                    labels_need_to_remove.remove(ans)
                except:
                    pass
        labels_need_to_remove_total += labels_need_to_remove
    labels_need_to_remove_total = set(labels_need_to_remove_total)
    print('total number of labels need to remove:', len(labels_need_to_remove_total))
    for ans in labels_need_to_remove_total:
        print(ans, label_count[ans])

    # remove labels from label2ans
    for ans in label2ans:
        if ans in labels_need_to_remove_total:
            label2ans.remove(ans)
    ans2label = {ans: i for i, ans in enumerate(label2ans)}

    # remove labels from splits
    splits = [train, val, test]
    for k, split in enumerate(splits):
        mask = np.ones(len(split), dtype=bool)
        for i in tqdm(range(len(split))):
            split[i]['answer']['labels'] = []
            # remove low freq labels ans answers
            for ans in split[i]['answer']['answer']:
                if ans in labels_need_to_remove_total:
                    split[i]['answer']['answer'].remove(ans)
                    # split[i]['answer']['labels'].remove(ans2label[ans])
                    split[i]['answer']['scores'].pop()
                    if split[i]['answer']['answer'] == []:
                        # remove this record
                        mask[i] = False
            # reassign new labels
            for ans in split[i]['answer']['answer']:
                split[i]['answer']['labels'].append(ans2label[ans])
        # sample list by mask
        splits[k] = [split[i] for i in range(len(split)) if mask[i]]



    # reassign labels to splits
    for k, split in enumerate(splits):
        for i in range(len(split)):
            labels = []
            for ans in split[i]['answer']['answer']:
                labels.append(ans2label[ans])
            split[i]['answer']['labels'] = labels

    train, val, test = splits
    print('train:', len(train))
    print('val:', len(val))
    print('test:', len(test))

    # save files
    with open(label2ans_path, 'wb') as f:
        pickle.dump(label2ans, f)
    with open(ans2label_path, 'wb') as f:
        pickle.dump(ans2label, f)
    with open(train_path, 'wb') as f:
        pickle.dump(train, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test, f)

def remove_difference_from_question_paris(remove_temp_files = False):
    path = ['temp/mimic_dataset_train.pkl', 'temp/mimic_dataset_val.pkl', 'temp/mimic_dataset_test.pkl']
    new_dataset = []
    for p in path:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        for i in range(len(data)):
            record = {'subject_id': data[i]['subject_id'],
                      'study_id': data[i]['study_id'],
                      'question_type': data[i]['question_type'],
                      'question': data[i]['question'],
                      'answer': data[i]['answer']['answer'],
                      'split': p.split('_')[-1][:-4]
                      }
            new_dataset.append(record)
    # save to csv file
    df = pd.DataFrame(new_dataset)
    df.to_csv('../mimic_vqa_pairs.csv', index=False)

    # remove tempory files
    if remove_temp_files:
        files = os.listdir('temp')
        for f in files:
            os.remove('temp/' + f)

def main():
    preprocess_dataset()
    for i in range(10):
        remove_low_freq_labels()
    remove_difference_from_question_paris(remove_temp_files=False)


if __name__ == '__main__':
    main()