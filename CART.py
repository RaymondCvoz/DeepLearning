from math import log
    
# �������ݼ�
def create_dataset():
    dataset = [['youth', 'no', 'no', 'just so-so', 'no'],
               ['youth', 'no', 'no', 'good', 'no'],
               ['youth', 'yes', 'no', 'good', 'yes'],
               ['youth', 'yes', 'yes', 'just so-so', 'yes'],
               ['youth', 'no', 'no', 'just so-so', 'no'],
               ['midlife', 'no', 'no', 'just so-so', 'no'],
               ['midlife', 'no', 'no', 'good', 'no'],
               ['midlife', 'yes', 'yes', 'good', 'yes'],
               ['midlife', 'no', 'yes', 'great', 'yes'],
               ['midlife', 'no', 'yes', 'great', 'yes'],
               ['geriatric', 'no', 'yes', 'great', 'yes'],
               ['geriatric', 'no', 'yes', 'good', 'yes'],
               ['geriatric', 'yes', 'no', 'good', 'yes'],
               ['geriatric', 'yes', 'no', 'great', 'yes'],
               ['geriatric', 'no', 'no', 'just so-so', 'no']]
    features = ['age', 'work', 'house', 'credit']
    return dataset, features

# ���㵱ǰ���ϵ�Giniϵ��
def calcGini(dataset):
    # ����������
    num_of_examples = len(dataset)
    labelCnt = {}
    # ����������������
    for example in dataset:
        # ��ǰ�����ı�ǩֵ�Ǹ��б�����һ��Ԫ��
        currentLabel = example[-1]
        # ͳ��ÿ����ǩ�������˼���
        if currentLabel not in labelCnt.keys():
            labelCnt[currentLabel] = 0
        labelCnt[currentLabel] += 1
    # �õ��˵�ǰ������ÿ����ǩ�����������󣬼������ǵ�pֵ
    for key in labelCnt:
        labelCnt[key] /= num_of_examples
        labelCnt[key] = labelCnt[key] * labelCnt[key]
    # ����Giniϵ��
    Gini = 1 - sum(labelCnt.values())
    return Gini
    
# ��ȡ�Ӽ���
# ���ܣ���dataSet�����ҵ����е�axis����ǩֵ = value������
# Ȼ����Щ����ɾȥ��axis����ǩֵ����ȫ����ȡ������Ϊһ���µ�������
def create_sub_dataset(dataset, index, value):
    sub_dataset = []
    for example in dataset:
        current_list = []
        if example[index] == value:
            current_list = example[:index]
            current_list.extend(example[index + 1 :])
            sub_dataset.append(current_list)
    return sub_dataset

# ����ǰ�������ָ������iȡֵΪvalue��һ���ֺ�ȡֵ��Ϊvalue��һ���֣����֣�
def split_dataset(dataset, index, value):
    sub_dataset1 = []
    sub_dataset2 = []
    for example in dataset:
        current_list = []
        if example[index] == value:
            current_list = example[:index]
            current_list.extend(example[index + 1 :])
            sub_dataset1.append(current_list)
        else:
            current_list = example[:index]
            current_list.extend(example[index + 1 :])
            sub_dataset2.append(current_list)
    return sub_dataset1, sub_dataset2

def choose_best_feature(dataset):
    # ��������
    numFeatures = len(dataset[0]) - 1
    # ��ֻ��һ������ʱ
    if numFeatures == 1:
        return 0
    # ��ʼ����ѻ���ϵ��
    bestGini = 1
    # ��ʼ����������
    index_of_best_feature = -1
    # ��������������Ѱ�����������͸������µ������зֵ�
    for i in range(numFeatures):
        # ȥ�أ�ÿ������ֵΨһ
        uniqueVals = set(example[i] for example in dataset)
        # Gini�ֵ��е�ÿ��ֵ�����Ը�ֵ��Ӧ�ļ���Ϊ�зֵ�Ե�ǰ���Ͻ��л��ֺ��Giniϵ��
        Gini = {}
        # ���ڵ�ǰ������ÿ��ȡֵ
        for value in uniqueVals:
            # �����ɸ�ֵ���л��ֵõ��������Ӽ�
            sub_dataset1, sub_dataset2 = split_dataset(dataset,i,value)
            # �������Ӽ�ռԭ���ϵı���ϵ��prob1 prob2
            prob1 = len(sub_dataset1) / float(len(dataset))
            prob2 = len(sub_dataset2) / float(len(dataset))
            # �����Ӽ�1��Giniϵ��
            Gini_of_sub_dataset1 = calcGini(sub_dataset1)
            # �����Ӽ�2��Giniϵ��
            Gini_of_sub_dataset2 = calcGini(sub_dataset2)
            # �����ɵ�ǰ�����зֵ㻮�ֺ������Giniϵ��
            Gini[value] = prob1 * Gini_of_sub_dataset1 + prob2 * Gini_of_sub_dataset2
            # �������������������зֵ�
            if Gini[value] < bestGini:
                bestGini = Gini[value]
                index_of_best_feature = i
                best_split_point = value
    return index_of_best_feature, best_split_point
    
# ���ؾ���������������Ǹ���ǩ��ֵ��'yes' or 'no'��
def find_label(classList):
    # ��ʼ��ͳ�Ƹ���ǩ�������ֵ�
    # ��Ϊ����ǩ����Ӧ��ֵΪ��ǩ���ֵĴ���
    labelCnt = {}
    for key in classList:
        if key not in labelCnt.keys():
            labelCnt[key] = 0
        labelCnt[key] += 1
    # ��classCount��ֵ��������
    # ���磺sorted_labelCnt = {'yes': 9, 'no': 6}
    sorted_labelCnt = sorted(labelCnt.items(), key = lambda a:a[1], reverse = True)
    # ��������д��������
    # sortedClassCount = sorted(labelCnt.iteritems(), key=operator.itemgetter(1), reverse=True)
    # ȡsorted_labelCnt�е�һ��Ԫ���еĵ�һ��ֵ����Ϊ����
    return sorted_labelCnt[0][0]
    
    
def create_decision_tree(dataset, features):
    # ���ѵ�������������ı�ǩ
    # ���ڳ�ʼ���ݼ�����label_list = ['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    label_list = [example[-1] for example in dataset]
    # ��д�����ݹ�����������
    # ����ǰ���ϵ�����������ǩ��ȣ��������ѱ��֡�������
    # ��ֱ�ӷ��ظñ�ǩֵ��Ϊһ��Ҷ�ӽڵ�
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # ��ѵ������������������ʹ����ϣ���ǰ�޿�����������������δ���֡�����
    # �򷵻������������ı�ǩ��Ϊ���
    if len(dataset[0]) == 1:
        return find_label(label_list)
    # ��������ʽ�����Ĺ���
    # ѡȡ���з�֧������������±������зֵ�
    index_of_best_feature, best_split_point = choose_best_feature(dataset)
    # �õ��������
    best_feature = features[index_of_best_feature]
    # ��ʼ��������
    decision_tree = {best_feature: {}}
    # ʹ�ù���ǰ�����������ɾȥ
    del(features[index_of_best_feature])
    # ������ = ��ǰ��������Ϊ�ղ��Ѿ�ɾȥ���ù���������
    sub_labels = features[:]
    # �ݹ����create_decision_treeȥ�����½ڵ�
    # �����������зֵ㻮�ֳ����Ķ����Ӽ�
    sub_dataset1, sub_dataset2 = split_dataset(dataset,index_of_best_feature,best_split_point)
    # ����������
    decision_tree[best_feature][best_split_point] = create_decision_tree(sub_dataset1, sub_labels)
    # ����������
    decision_tree[best_feature]['others'] = create_decision_tree(sub_dataset2, sub_labels)
    return decision_tree
    
# ������ѵ���õľ�����������������
def classify(decision_tree, features, test_example):
    # ���ڵ���������
    first_feature = list(decision_tree.keys())[0]
    # second_dict�ǵ�һ���������Ե�ֵ��Ҳ���ֵ䣩
    second_dict = decision_tree[first_feature]
    # ������������ԣ��������Ա�ǩ�е�λ�ã����ڼ�������
    index_of_first_feature = features.index(first_feature)
    # ����second_dict�е�ÿһ��key
    for key in second_dict.keys():
        # ������'others'��key
        if key != 'others':
            if test_example[index_of_first_feature] == key:
            # ����ǰsecond_dict��key��value��һ���ֵ�
                if type(second_dict[key]).__name__ == 'dict':
                    # ����Ҫ�ݹ��ѯ
                    classLabel = classify(second_dict[key], features, test_example)
                # ����ǰsecond_dict��key��value��һ��������ֵ
                else:
                    # �����Ҫ�ҵı�ǩֵ
                    classLabel = second_dict[key]
            # ������������ڵ�ǰ������ȡֵ������key����˵�����ڵ�ǰ������ȡֵ����'others'
            else:
                # ���second_dict['others']��ֵ�Ǹ��ַ�������ֱ�����
                if isinstance(second_dict['others'],str):
                    classLabel = second_dict['others']
                # ���second_dict['others']��ֵ�Ǹ��ֵ䣬��ݹ��ѯ
                else:
                    classLabel = classify(second_dict['others'], features, test_example)
    return classLabel
    
if __name__ == '__main__':
    dataset, features = create_dataset()
    decision_tree = create_decision_tree(dataset, features)
    # ��ӡ���ɵľ�����
    print(decision_tree)
    # �����������з������
    features = ['age', 'work', 'house', 'credit']
    test_example = ['midlife', 'yes', 'no', 'great']
    print(classify(decision_tree, features, test_example))
