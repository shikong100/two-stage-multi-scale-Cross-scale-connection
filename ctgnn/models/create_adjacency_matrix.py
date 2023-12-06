import os
import pandas as pd
import numpy as np
import argparse

DefectLabels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]
WaterLabels = ["0%<5%","5-15%","15-30%","30%<="]
      
waterIntervals = [5, 15, 30]


def create_cooccurrence_matrix(annRoot, split):
    gtPath = os.path.join(annRoot, "SewerML_{}.csv".format(split))
    gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["WaterLevel"] + DefectLabels)
    # Defect labels
    labels_defects = gt[DefectLabels].values 
    print(f'labels_defects->{labels_defects}')
    num_classes_defects = len(DefectLabels)

    # Water labels
    labels_water = gt["WaterLevel"].values # 重新设置water的标签,分为4个等级

    if len(waterIntervals) > 0:
        num_classes_water = len(waterIntervals)+1 # num_classes_water = 4
        labels_water[labels_water < waterIntervals[0]] = 0 # 小于5设置为0
        labels_water[labels_water >= waterIntervals[-1]] = num_classes_water-1 # 大于30设置为3
        for idx in range(1, len(waterIntervals)): # 大于5小于15设置为1，大于15小于30设置为2
            labels_water[(labels_water >= waterIntervals[idx-1]) & (labels_water < waterIntervals[idx])] = idx
    else:
        uniqueLevels = np.unique(labels_water)
        num_classes_water = len(uniqueLevels)
        for idx, level in enumerate(uniqueLevels):
            labels_water[labels_water == level] = idx


    num_classes_total = num_classes_defects + num_classes_water # 等于17+4 = 21
    print("#Defects: {}\n#Water: {}\nTotal: {}".format(num_classes_defects, num_classes_water, num_classes_total))

    # 共现矩阵
    cooccurrence_matrix = np.zeros((num_classes_total, num_classes_total))
    class_sum = np.zeros((num_classes_total), dtype=np.int)

    for idx_1 in range(num_classes_total):
        if idx_1 < num_classes_defects:
            class_labels_idx1 = labels_defects[:,idx_1]
        elif idx_1 < (num_classes_defects + num_classes_water):
            class_labels_idx1 = labels_water == (idx_1 - num_classes_defects)

        class_sum[idx_1] = np.sum(class_labels_idx1)
        for idx_2 in range(idx_1, num_classes_total):
            if idx_2 < num_classes_defects:
                class_labels_idx2 = labels_defects[:,idx_2]
            elif idx_2 < (num_classes_defects + num_classes_water):
                class_labels_idx2 = labels_water == (idx_2 - num_classes_defects)

            co_occurrence = np.logical_and(class_labels_idx1, class_labels_idx2)

            cooccurrence_matrix[idx_1, idx_2] = cooccurrence_matrix[idx_2, idx_1] = np.sum(co_occurrence)

    return cooccurrence_matrix, class_sum, [num_classes_defects, num_classes_water]


def get_probability_matrix(coocc, class_count, task_num_classes):
    
    prob = np.zeros_like(coocc) # (21, 21)
    task_indecies = []
    for task_idx in range(len(task_num_classes)+1): # range(3)
        task_indecies.append(sum(task_num_classes[:task_idx])) # [0, 17, 21]
    
    # Calculate probability of row element given the column class
    for task_idx1 in range(len(task_num_classes)): # range(2)
        for task_idx2 in range(len(task_num_classes)): # range(2)
            sub_coocc = coocc[task_indecies[task_idx1]:task_indecies[task_idx1+1], task_indecies[task_idx2]:task_indecies[task_idx2+1]]

            sub_prob = sub_coocc / class_count[task_indecies[task_idx2]:task_indecies[task_idx2+1]]

            prob[task_indecies[task_idx1]:task_indecies[task_idx1+1], task_indecies[task_idx2]:task_indecies[task_idx2+1]] = sub_prob
    return prob


def get_binarized_matrix(prob, task_num_classes, binary_threshold = None):
    if binary_threshold == None:
        binary_threshold = [1/num_classes for num_classes in task_num_classes]
        print("Using uniform random class probability as binary threshold")
    else:
        print("Using user provided binary threshold")
    
    if isinstance(binary_threshold, (float, int)):
        binary_threshold = [binary_threshold for _ in task_num_classes]
        
    binary = np.zeros_like(prob) # (21, 21)
    task_indecies = []
    for task_idx in range(len(task_num_classes)+1):
        task_indecies.append(sum(task_num_classes[:task_idx])) # [0, 17, 21]


    for task_idx1 in range(len(task_num_classes)): # range(2)
        for task_idx2 in range(len(task_num_classes)): # range(2)
            sub_binary = prob[task_indecies[task_idx1]:task_indecies[task_idx1+1], task_indecies[task_idx2]:task_indecies[task_idx2+1]].copy()

            sub_binary[sub_binary < binary_threshold[task_idx1]] = 0
            sub_binary[sub_binary >= binary_threshold[task_idx1]] = 1

            binary[task_indecies[task_idx1]:task_indecies[task_idx1+1], task_indecies[task_idx2]:task_indecies[task_idx2+1]] = sub_binary

    return binary


def get_reweighted_matrix(binary, neighbour_weight):
    reweighted = np.zeros_like(binary)

    binary = binary - np.identity(binary.shape[0]) # Remove the self loop to make neighbour weight easier to calculate

    # s = binary.sum(1, keepdims=True)
    # print(f's->{s}')
    # reweighted = binary * neighbour_weight / s
    reweighted = binary * neighbour_weight / binary.sum(1, keepdims=True) 
    reweighted = np.where(np.isnan(reweighted), 0.0, reweighted)
    reweighted = reweighted + np.identity(reweighted.shape[0], np.int)*(1-neighbour_weight)
    
    row_sums = np.sum(reweighted, axis=1)
    for idx in range(reweighted.shape[0]):
        if row_sums[idx] < 0.95: # Hack value to ensure that the sum is below 1, but allow for floating point error
            reweighted[idx,idx] = 1.0


    return reweighted



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', type=str, default="/mnt/data0/qh/Sewer/annotations")
    parser.add_argument('--outputPath', type=str, default="./adjacency_matrices")
    parser.add_argument('--split', type=str, default = "Train")
    parser.add_argument('--binary_threshold', type=float, default = 0.05)
    parser.add_argument('--neighbour_weight', type=float, default = 0.2)
    parser.add_argument('--valid_tasks', nargs='+', type=str, default = ["defect", "water"])
    args = parser.parse_args()

    task_LUT = {"defect": 0,
                "water": 1}
    valid_tasks = args.valid_tasks

    args = vars(args)

    outputPath = args["outputPath"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    coocc, class_sum, task_num_classes = create_cooccurrence_matrix(args["inputPath"], args["split"]) # 共现矩阵，每个类总数，[17, 4]

    ####### 开始 ------------------------------------------------------------------------------------------
    task_indecies = [task_LUT[task] for task in valid_tasks] # [0, 1]
    task_num_classes_tmp = [task_num_classes[t_idx] for t_idx in task_indecies] # [17, 4]
    cooc_tmp = np.zeros((sum(task_num_classes_tmp), sum(task_num_classes_tmp))) # (21, 21)
    mask = np.zeros_like(coocc) # (21, 21)
    classSum_tmp = np.zeros(sum(task_num_classes_tmp), dtype=np.int) # (21, )

    for idx1, task_class1 in enumerate(task_num_classes):
        new_start = 0
        new_end = -1
        for idx2, task_class2 in enumerate(task_num_classes):
            if idx1 not in task_indecies:
                continue
            if idx2 not in task_indecies:
                continue

            start_idx1 = sum(task_num_classes[:idx1])
            end_idx1 = sum(task_num_classes[:idx1+1])
            start_idx2 = sum(task_num_classes[:idx2])
            end_idx2 = sum(task_num_classes[:idx2+1])

            mask[start_idx1:end_idx1, start_idx2:end_idx2] = 1

            new_end = new_start + task_class2
            classSum_tmp[new_start:new_end] = class_sum[start_idx2:end_idx2]
            new_start = new_end     

    mask = mask.astype(np.bool) # 全1，astype后为全True
    cooc_tmp = np.ravel(coocc)[np.ravel(mask)] # 拉伸为1维

    # reshape the squished array to a 2D square
    cooc_tmp = cooc_tmp.reshape(                         # reshape the array
            np.sqrt(cooc_tmp.size).astype(int),     # pass to int to stop warning
            -1                                      # fill next axis with leftover
        )
    
    coocc = cooc_tmp

    task_num_classes = task_num_classes_tmp # 没有改变
    class_sum = classSum_tmp # 没有改变
    ### 结束，中间没毛用，妈的
    #-----------------------------------------------------------------------------------------------------------------
    prob = get_probability_matrix(coocc, class_sum, task_num_classes)
    binary = get_binarized_matrix(prob, task_num_classes, args["binary_threshold"])
    reweighted = get_reweighted_matrix(binary, args["neighbour_weight"])

    prob_masked = prob * binary
    reweighted_cond = get_reweighted_matrix(prob_masked, args["neighbour_weight"])

    full_binary = np.ones_like(reweighted_cond)

    np.save(os.path.join(outputPath, "co_occurrence.npy"), coocc)
    np.save(os.path.join(outputPath, "cond_probability.npy"), prob)
    np.save(os.path.join(outputPath, "adj_binary.npy"), binary)
    np.save(os.path.join(outputPath, "adj_reweighted.npy"), reweighted)
    np.save(os.path.join(outputPath, "adj_reweightedCond.npy"), reweighted_cond)
    np.save(os.path.join(outputPath, "adj_binaryFull.npy"), full_binary)

    
    np.savetxt(os.path.join(outputPath, "co_occurrence.csv"), coocc, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(outputPath, "cond_probability.csv"), prob, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(outputPath, "adj_binary.csv"), binary, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(outputPath, "adj_reweighted.csv"), reweighted, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(outputPath, "adj_reweightedCond.csv"), reweighted_cond, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(outputPath, "adj_binaryFull.csv"), full_binary, delimiter=",", fmt='%f')
