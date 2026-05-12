import math

import numpy as np
import sklearn.metrics as metrics
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from shapely.geometry import Polygon
# from skimage.segmentation import mark_boundaries
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


def get_class(img_index, index):
    try:
        if index[0] < 0 or index[1] < 0:
            return -1
        class_num = img_index[index[0]][index[1]]
        return class_num
    except:
        return -1


def linear_detection(img_index, center_index, direction):
    center_class = get_class(img_index, center_index)
    neighbour_class = np.copy(center_class)
    detection_index = np.copy(center_index)
    while neighbour_class == center_class:
        if neighbour_class == -1:
            break
        detection_index += direction
        neighbour_class = get_class(img_index, detection_index)
    return neighbour_class


def pca(num, img_data):
    pca = PCA(n_components=num, whiten=True)
    new_data = img_data.T
    new_data = pca.fit_transform(new_data)
    return new_data


def get_neighbour(img_index, center_index):
    direction = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [0, -1],
                 [1, 1], [1, 0], [1, -1]]
    neighbour_class = np.zeros(1)
    for item in direction:
        item_np = np.array(item)
        neighbour_class = np.append(neighbour_class,
                                    linear_detection(img_index, center_index, item_np))
    neighbour_class = np.delete(neighbour_class, 0)
    return neighbour_class


def construct_patch(imgIndex, idxNum, img, img_pca, target_spectral, threshold, target_patch_num_train,
                    target_patch_num_test,
                    background_patch_num_train, background_patch_num_test):
    cluster_data, cluster_data_pca, cluster_cem_data = get_cluster(idxNum, imgIndex, img, img_pca, target_spectral)
    num = len(cluster_data)
    center_index = get_centroid(idxNum, imgIndex)
    border_num = getInnerBorderNum(idxNum, imgIndex)
    spectral_feature_pca = np.zeros((img_pca.shape[0], 1))
    spectral_feature = np.zeros(target_spectral.shape)
    cem_feature = np.zeros(1)
    target_cluster_num = 0
    background_cluster_num = 0
    border = getOuterBorder(idxNum, imgIndex)

    # 获取特征光谱 和每个超像素的cem均值
    for i in range(num):
        temp_spectral = np.sum(cluster_data[i][0], axis=1)
        spectral_feature = np.append(spectral_feature, (temp_spectral / cluster_data[i][0].shape[1]).reshape(-1, 1),
                                     axis=1)
        temp_spectral_pca = np.sum(cluster_data_pca[i][0], axis=1)
        spectral_feature_pca = np.append(spectral_feature_pca,
                                         (temp_spectral_pca / cluster_data_pca[i][0].shape[1]).reshape(-1, 1),
                                         axis=1)
        temp_cem = np.sum(cluster_cem_data[i])
        cem_feature = np.append(cem_feature, temp_cem / cluster_cem_data[i][0].shape[0])
    spectral_feature = np.delete(spectral_feature, 0, axis=1)
    spectral_feature_pca = np.delete(spectral_feature_pca, 0, axis=1)
    cem_feature = np.delete(cem_feature, 0)

    # 粗划分目标背景
    target_cluster_index = []
    for i in range(spectral_feature.shape[1]):
        if judge_target_cluster(spectral_feature[..., i], target_spectral, threshold, cem_feature[i]):
            target_cluster_num += cluster_data[i][0].shape[1]
            target_cluster_index.append(i)
        else:
            background_cluster_num += cluster_data[i][0].shape[1]

    # 粗划分目标测试图
    coarse_target_img = np.zeros((img.shape[1], img.shape[2]))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            cluster_number = np.copy(imgIndex[i][j]).tolist()
            if cluster_number in target_cluster_index:
                coarse_target_img[i][j] = 1
    plt.imshow(coarse_target_img)
    plt.show()

    # 构建目标，背景patch
    background_patch_train, target_patch_train, background_center_spectral_train, target_center_spectral_train = construction(
        img, img_pca, center_index, imgIndex, spectral_feature, spectral_feature_pca, cluster_data, cluster_data_pca,
        target_spectral, threshold, target_cluster_num, border,
        background_cluster_num, border_num, background_patch_num_train, target_patch_num_train, cem_feature)
    background_patch_test, target_patch_test, background_center_spectral_test, target_center_spectral_test = construction(
        img, img_pca, center_index, imgIndex, spectral_feature, spectral_feature_pca, cluster_data, cluster_data_pca,
        target_spectral, threshold, target_cluster_num, border,
        background_cluster_num, border_num, background_patch_num_test, target_patch_num_test, cem_feature)
    return background_patch_train, target_patch_train, background_center_spectral_train, target_center_spectral_train, \
           background_patch_test, target_patch_test, background_center_spectral_test, target_center_spectral_test


def construction(img, img_pca, center_index, imgIndex, spectral_feature, spectral_feature_pca, cluster_data,
                 cluster_data_pca, target_spectral, threshold,
                 target_cluster_num, border, background_cluster_num, border_num, background_patch_num, target_patch_num,
                 cem_result):
    direction = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [0, -1],
                 [1, 1], [1, 0], [1, -1]]
    background_patch = np.zeros((1, img.shape[0], 3, 3))
    target_patch = np.zeros((1, img.shape[0], 3, 3))
    background_patch_pca = np.zeros((1, img_pca.shape[0], 3, 3))
    target_patch_pca = np.zeros((1, img_pca.shape[0], 3, 3))
    target_center_spectral = np.zeros((img.shape[0], 1))
    background_center_spectral = np.zeros((img.shape[0], 1))

    for i in range(0, len(center_index)):
        # neighbour_class_index = get_neighbour(imgIndex, center_index[i])
        cluster_feature_spectral = spectral_feature[..., i]
        cluster_feature_spectral_pca = spectral_feature_pca[..., i]
        cluster_spectral = cluster_data[i][0]
        cluster_spectral_pca = cluster_data_pca[i][0]

        neighbour_class_index = get_neighbour_arrangement(imgIndex, border[i], i)
        cluster_judge = judge_target_cluster(cluster_feature_spectral, target_spectral, threshold, cem_result[i])
        if cluster_judge:
            cluster_num = target_cluster_num
            cluster_feature_spectral = np.copy(target_spectral.reshape(-1))
            patch_num = target_patch_num
        else:
            cluster_num = background_cluster_num
            patch_num = background_patch_num
        for j in range(0, int(math.ceil(patch_num * cluster_spectral.shape[1] / cluster_num))):
            '''
            加入构造概率 随机构造带边不带边
            '''
            neighbour_class_random_choice = np.random.randint(0, neighbour_class_index.shape[0])
            temp_patch = np.zeros((1, img.shape[0], 3, 3))
            temp_patch_pca = np.zeros((1, img_pca.shape[0], 3, 3))
            '''
            中心像素为类的特征光谱与类内光谱按照一定比例相加
            '''
            random_add_percentage = np.random.randint(20, 30)  # 光谱相加权重，构造带噪声的光谱，增加鲁棒性
            random_add_inner_spectral_index = np.random.randint(0, cluster_spectral.shape[1])
            center_spectral = (random_add_percentage * cluster_feature_spectral + (
                    100 - random_add_percentage) * cluster_spectral[
                                   ..., random_add_inner_spectral_index]) / 100
            temp_patch[0, ..., 1, 1] = center_spectral
            # PCA data
            center_spectral_pca = (random_add_percentage * cluster_feature_spectral_pca + (
                    100 - random_add_percentage) * cluster_spectral_pca[
                                       ..., random_add_inner_spectral_index]) / 100
            temp_patch_pca[0, ..., 1, 1] = center_spectral_pca
            # 是否带边
            random_choice_percentage = np.random.randint(1, 100) / 100
            # neighbour_class = [[]] # 构造相邻边数量的排列
            # 计算边界像素个数和类内像素个数的比值，如果小于这个比值则patch带边 否则patch不带边
            if random_choice_percentage < border_num[i] / cluster_spectral.shape[1]:
                neighbour_class = np.copy(neighbour_class_index[neighbour_class_random_choice, :, :].reshape(
                    [neighbour_class_index.shape[1], neighbour_class_index.shape[2]]))
                for k in range(len(direction)):
                    # if neighbour_class[k] != -1:
                    if neighbour_class[direction[k][0], direction[k][1]] != -1:
                        # 光谱相加权重，构造带噪声的光谱，增加鲁棒性
                        random_add_percentage = np.random.randint(60, 80)
                        # 内部光谱增加比例
                        random_add_inner_spectral_index = np.random.randint(0, cluster_spectral.shape[1])
                        # 边类型
                        random_add_neighbour = 1  # np.random.randint(0, 1)
                        # 带边则值为random_add_percentage 否则为0
                        random_percentage = random_add_neighbour * random_add_percentage
                        neighbour_spectral = (random_percentage * spectral_feature[
                            ..., int(neighbour_class[direction[k][0], direction[k][1]])] + (
                                                      100 - random_percentage) *
                                              cluster_spectral[..., random_add_inner_spectral_index]) / 100
                        temp_patch[0, ..., 1 + direction[k][0], 1 + direction[k][1]] = neighbour_spectral
                        # PCA
                        neighbour_spectral_pca = (random_percentage * spectral_feature_pca[
                            ..., int(neighbour_class[direction[k][0], direction[k][1]])] + (100 - random_percentage) *
                                                  cluster_spectral_pca[..., random_add_inner_spectral_index]) / 100
                        temp_patch_pca[0, ..., 1 + direction[k][0], 1 + direction[k][1]] = neighbour_spectral_pca
                    else:
                        continue
            else:
                for k in range(0, len(direction)):
                    # 光谱相加权重，构造带噪声的光谱，增加鲁棒性
                    random_add_percentage = np.random.randint(60, 80)
                    # 内部光谱增加比例
                    random_add_inner_spectral_index = np.random.randint(0, cluster_spectral.shape[1])
                    # 构造由内部光谱和特征光谱混合的混合像元
                    neighbour_spectral = (random_add_percentage * cluster_feature_spectral + (
                            100 - random_add_percentage) * cluster_spectral[
                                              ..., random_add_inner_spectral_index]) / 100
                    temp_patch[0, ..., 1 + direction[k][0], 1 + direction[k][1]] = neighbour_spectral
                    # PCA
                    neighbour_spectral_pca = (random_add_percentage * cluster_feature_spectral_pca + (
                            100 - random_add_percentage) * cluster_spectral_pca[
                                                  ..., random_add_inner_spectral_index]) / 100
                    temp_patch_pca[0, ..., 1 + direction[k][0], 1 + direction[k][1]] = neighbour_spectral_pca
            if cluster_judge:
                target_patch = np.append(target_patch, temp_patch, axis=0)
                target_patch_pca = np.append(target_patch_pca, temp_patch_pca, axis=0)
                target_center_spectral = np.append(target_center_spectral, center_spectral.reshape(-1, 1), axis=1)
            else:
                background_patch = np.append(background_patch, temp_patch, axis=0)
                background_patch_pca = np.append(background_patch_pca, temp_patch_pca, axis=0)
                background_center_spectral = np.append(background_center_spectral, center_spectral.reshape(-1, 1),
                                                       axis=1)
    background_patch = np.delete(background_patch, 0, axis=0)
    target_patch = np.delete(target_patch, 0, axis=0)
    background_center_spectral = np.delete(background_center_spectral, 0, axis=1)
    target_center_spectral = np.delete(target_center_spectral, 0, axis=1)
    return background_patch, target_patch, background_center_spectral, target_center_spectral  # background_patch_pca, target_patch_pca,


def get_neighbour_arrangement(imgIdx, border, class_idx):
    border_index = change_to_index(border)
    direction = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [0, -1],
                 [1, 1], [1, 0], [1, -1]]
    border_patch = np.zeros((1, 3, 3))
    for index in border_index:
        border_patch_temp = np.zeros((1, 3, 3))
        border_patch_temp[0, 1, 1] = class_idx
        for dir in direction:
            try:
                border_patch_temp[0, 1 + dir[0], 1 + dir[1]] = imgIdx[index[0] + dir[0], index[1] + dir[1]]
            except:
                border_patch_temp[0, 1 + dir[0], 1 + dir[1]] = -1
        border_patch = np.append(border_patch, border_patch_temp, axis=0)
    border_patch = np.delete(border_patch, 0, axis=0)
    border_patch = np.unique(border_patch, axis=0)
    return border_patch


# def showSuperpixel(hsi_img, segments):
#     numSegments = np.max(segments) + 1
#     image = hsi_img[10, :, :]
#     image = np.expand_dims(image, axis=2)
#     image = np.append(image, np.expand_dims(hsi_img[20, :, :], axis=2), axis=2)
#     image = np.append(image, np.expand_dims(hsi_img[30, :, :], axis=2), axis=2)
#     fig = plt.figure("Superpixels -- %d segments" % (numSegments))
#
#     ax1 = fig.add_subplot(121)
#     ax1.imshow(image, interpolation="none")
#
#     ax = fig.add_subplot(122)
#     cube = image
#     cube = (cube - np.min(cube)) / (np.max(cube) - np.min(cube))
#     ax.imshow(mark_boundaries(cube, segments, mode='subpixel'), interpolation="none")
#
#     spectral.save_rgb("test.png", mark_boundaries(cube[:, :, [0, 1, 2]], segments), colors=spectral.spy_colors)
#     plt.pause(10)
#
#     # show the plots
#     plt.show()


def getOuterBorder(idxNum, imgIdx):
    border = []
    binary_map = getBinaryMap(idxNum, imgIdx)
    for i in range(idxNum):
        map = binary_map[i].reshape(imgIdx.shape[0] + 2, imgIdx.shape[1] + 2)
        # erosion_map_temp = ndimage.binary_erosion(map).astype(map.dtype)
        dilation_map = ndimage.binary_dilation(map).astype(map.dtype)
        outter_border = dilation_map - binary_map[i]  # erosion_map
        border.append(outter_border)
    return border


def getInnerBorderNum(idxNum, imgIdx):
    num = []
    binary_map = getBinaryMap(idxNum, imgIdx)
    for i in range(idxNum):
        map = binary_map[i].reshape(imgIdx.shape[0] + 2, imgIdx.shape[1] + 2)
        erosion_map_temp = ndimage.binary_erosion(map).astype(map.dtype)
        inner_border = erosion_map_temp - binary_map[i]  # erosion_map
        num.append(np.count_nonzero(inner_border))
    return num


def getBinaryMap(idxNum, imgIdx):
    binary_map = np.zeros((idxNum, imgIdx.shape[0] + 2, imgIdx.shape[1] + 2))
    for i in range(imgIdx.shape[0]):
        for j in range(imgIdx.shape[1]):
            binary_map[imgIdx[i, j], i + 1, j + 1] = 1
    return binary_map


def change_to_index(border):
    # direction = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1]]
    dir = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    index = []
    for i in range(border.shape[0]):
        for j in range(border.shape[1]):
            if border[i][j] == 1:
                start_index = [i, j]
                break
        # if border[i][j] == 1:
        #     break
    temp_index = start_index
    while temp_index not in index:
        index.append(temp_index)
        for direction in dir:
            id = [index[-1][0] + direction[0], index[-1][1] + direction[1]]
            try:
                if border[id[0]][id[1]] == 1 and id in index:
                    continue
                elif border[id[0]][id[1]] == 1:
                    temp_index = id
                    break
            except:
                continue
    index_np = np.array(index)
    return index_np


def bubble_sort_flag(list, axis):  # 根据数组的某一维排序
    length = len(list)
    for index in range(length):
        # 标志位
        flag = True
        for j in range(1, length - index):
            if axis == 0 and list[j - 1][0] > list[j][0]:
                list[j - 1][0], list[j][0] = list[j][0], list[j - 1][0]
                list[j - 1][1], list[j][1] = list[j][1], list[j - 1][1]
                flag = False
            if axis == 1 and list[j - 1][1] > list[j][1]:
                list[j - 1][0], list[j][0] = list[j][0], list[j - 1][0]
                list[j - 1][1], list[j][1] = list[j][1], list[j - 1][1]
                flag = False
        if flag:
            # 没有发生交换，直接返回list
            return list
    return list


def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]

    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)

    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)

    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def get_node(border):
    length = len(border)
    ang = []
    node = np.array([[-1, -1]])
    for p in range(0, length):
        temp_1 = np.array([border[(p + 1) % length][0], border[(p + 1) % length][1], border[p][0], border[p][1]])
        temp_2 = np.array([border[(p + 1) % length][0], border[(p + 1) % length][1], border[(p + 2) % length][0],
                           border[(p + 2) % length][1]])
        ang.append([angle(temp_1, temp_2), (p + 1) % length])
    ang_np = np.array(ang)
    ang_np = bubble_sort_flag(ang_np, 0)  # 根据数组的第一维排序
    for i in range(ang_np.shape[0]):
        if ang_np[i][0] < 180:
            node = np.append(node, border[i].reshape((1, 2)), axis=0)
        else:
            break
    node = np.delete(node, 0, axis=0)
    return node


def sam(r, d):
    rr = np.sum(r ** 2, 1) ** 0.5
    dd = np.sum(d ** 2, 1) ** 0.5
    rd = np.sum(r * d, 1)
    result = np.arccos(rd / (rr * dd))
    return result


'''
获取重心
'''


def get_centroid(idxNum, imgIdx):
    border = getOuterBorder(idxNum, imgIdx)
    centroid = []
    for i in range(len(border)):
        index = change_to_index(border[i])
        node = get_node(index)
        node_list = node.tolist()
        centroid_float = list(Polygon(node_list).centroid.coords)
        centroid_int = [round(float(centroid_float[0][0])), round(float(centroid_float[0][1]))]
        centroid.append(centroid_int)
    return centroid


def get_cluster(clusters, img_index, hsi_img, img_pca, target_spectral):
    result_data_temp = [[] for i in range(clusters)]
    result_data = [[] for i in range(clusters)]
    result_data_temp_pca = [[] for i in range(clusters)]
    result_data_pca = [[] for i in range(clusters)]
    cem_result_temp = [[] for i in range(clusters)]
    cem_result = [[] for i in range(clusters)]
    cem_all = cem(hsi_img, target_spectral)
    plt.imshow(cem_all.reshape(hsi_img.shape[1], hsi_img.shape[2]))
    plt.show()
    for i in range(img_index.shape[0]):
        for j in range(img_index.shape[1]):
            cluster_num = img_index[i][j]
            result_data_temp[cluster_num].append(hsi_img[..., i, j])
            result_data_temp_pca[cluster_num].append(img_pca[..., i, j])
            cem_result_temp[cluster_num].append(cem_all[i * hsi_img.shape[1] + j])
    for i in range(len(result_data_temp)):
        result_data[i].append(np.array(result_data_temp[i]).T)
        result_data_pca[i].append(np.array(result_data_temp_pca[i]).T)
        cem_result[i].append(np.array(cem_result_temp[i]))
    return result_data, result_data_pca, cem_result


def judge_target_cluster(spectral_feature, target_spectral, threshold, cem_result):
    p_1 = spectral_feature.reshape(-1, 1)
    sid_similar = sid(p_1, target_spectral)
    similar = (1 - sid_similar) * cem_result
    if similar < threshold:
        return False
    else:
        return True


def sid(p_1, p_2):
    r = p_2
    d = p_1
    m = r / np.sum(r, 0)
    n = d / np.sum(d, 0)
    drd = np.sum(m * np.log(m / n), 0)
    ddr = np.sum(n * np.log(n / m), 0)
    result = drd + ddr
    return result



def cem(hsi_img, tgt):
    target_spectral = np.copy(tgt)
    img = np.copy(hsi_img).reshape((-1, hsi_img.shape[1] * hsi_img.shape[2]))
    size = img.shape  # get the size of image matrix
    R = np.dot(img, img.T / size[1])  # R = X*X'/size(X,2);
    w = np.dot(np.linalg.inv(R), target_spectral)  # w = (R+lamda*eye(size(X,1)))\d ;
    result = np.dot(w.T, img).T  # y=w'* X;
    return result


class get_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        data_tensor = torch.from_numpy(self.data[item, :, :, :].reshape(1, -1, 3, 3))
        label_tensor = torch.from_numpy(self.label[:, item])
        return data_tensor, label_tensor

    def __len__(self):
        return self.label.shape[1]


class get_dataset_test(Dataset):
    def __init__(self, test_data_patch, target_patch, label):
        self.data = test_data_patch
        self.target = target_patch
        self.label = label

    def __getitem__(self, item):
        data_tensor = torch.from_numpy(self.data[item, :, :, :].reshape(1, -1, 3, 3))
        target_tensor = torch.from_numpy(self.target[item, :, :, :].reshape(1, -1, 3, 3))
        label = torch.from_numpy(self.label[:, item])
        return data_tensor, target_tensor, label

    def __len__(self):
        return self.data.shape[0]


def construct_test_img(hsi_img, target_spectral):
    direction = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [0, -1],
                 [1, 1], [1, 0], [1, -1]]
    test_patch = np.zeros([hsi_img.shape[1] * hsi_img.shape[2], 1, hsi_img.shape[0], 3, 3])
    test_target = np.copy(target_spectral).reshape(-1)
    test_target_patch = np.zeros([hsi_img.shape[1] * hsi_img.shape[2], 1, hsi_img.shape[0], 3, 3])
    test_label = np.zeros((hsi_img.shape[0], hsi_img.shape[1] * hsi_img.shape[2]))
    for i in range(0, hsi_img.shape[1]):
        for j in range(0, hsi_img.shape[2]):
            test_patch[i * hsi_img.shape[2] + j, 0, ..., 1, 1] = hsi_img[..., i, j]
            test_label[..., i * hsi_img.shape[2] + j] = hsi_img[..., i, j]
            test_target_patch[i * hsi_img.shape[2] + j, 0, ..., 1, 1] = test_target
            for dir in direction:
                if i + dir[0] < 0 or j + dir[1] < 0 or i + dir[0] >= hsi_img.shape[1] or j + dir[1] >= hsi_img.shape[2]:
                    continue
                else:
                    test_patch[i * hsi_img.shape[2] + j, 0, ..., 1 + dir[0], 1 + dir[1]] = hsi_img[
                        ..., i + dir[0], j + dir[1]]
                    test_target_patch[i * hsi_img.shape[2] + j, 0, ..., 1 + dir[0], 1 + dir[1]] = hsi_img[
                        ..., i + dir[0], j + dir[1]]
    return test_patch, test_target_patch, test_label


def max_min_normalization(data_value):
    data_shape = data_value.shape
    data_shape_0 = data_shape[0]
    data_shape_1 = data_shape[1]
    new_data = np.zeros(shape=(data_shape_0, data_shape_1))
    data_col_min_values = np.min(data_value)
    data_col_max_values = np.max(data_value)
    for i in range(0, data_shape_0):
        for j in range(0, data_shape_1):
            new_data[i, j] = (data_value[i, j] - data_col_min_values) / (
                    data_col_max_values - data_col_min_values)
    return new_data


def background_control(x, lamda):
    return 1 - np.exp(-lamda * x)


def mean_anomalous_values(result, times):
    index_all = []
    for i in range(times):
        result_max_anomalous_values_index = np.argmax(result)
        max_index_x, max_index_y = get_max_index(result, result_max_anomalous_values_index)
        index_all.append([max_index_x, max_index_y])
        result[max_index_x][max_index_y] = -999
    max = np.max(result)
    for k in range(times):
        result[index_all[k][0]][index_all[k][1]] = np.copy(max)
    return result


def get_max_index(array, row_index):
    x = int(row_index / array.shape[0])
    y = int(row_index % array.shape[0])
    return x, y


class Detector(object):
    def __init__(self, img_data):
        self.data = img_data
        self.name = img_data.name
        self.img = img_data.img
        self.tgt = img_data.tgt
        self.grt = img_data.grt

    def show(self, results, names):
        imgshow = [self.img[1,].reshape(self.grt.shape, order='F'), self.grt]
        nameshow = ['image(first band)', 'groundtruth'] + names
        for item in results:
            imgshow.append(item.reshape(self.grt.shape, order='F'))
        k = math.ceil(len(imgshow) / 3) * 100 + 31
        for i in range(len(imgshow)):  # show image
            plt.subplot(k + i)
            plt.axis('off')
            plt.imshow(imgshow[i]) # , cmap='gray'
            plt.title(nameshow[i])
        img_name = 'result ' + self.name + '.png'
        plt.savefig(img_name)
        auc = plot_ROC(self.grt.reshape(-1, 1, order='F'), results, names, self.name)  # plot ROC curve
        auc2 = plot_ROC2(self.grt.reshape(-1, 1, order='F'), results, names, self.name)  # plot ROC curve2
        return auc


def plot_ROC(test_labels, resultall, name, dataset_name):
    plt.subplots(num='ROC curve', figsize=[10, 7])
    img_name = dataset_name + ' ROC Curve.png'
    tem = resultall
    auc = []
    for i in range(len(resultall)):
        fpr, tpr, thresholds = metrics.roc_curve(
            test_labels, resultall[i], pos_label=1)  # caculate False alarm rate and Probability of detection
        auc.append("%.5f" % metrics.auc(fpr, tpr))  # caculate AUC (Area Under the Curve)
        print('%s_AUC: %s' % (name[i][0], auc[i]))
        if not i: my_plot = plt.semilogx if metrics.auc(fpr, tpr) > 0.9 else plt.plot
        my_plot(fpr, tpr, label=name[i] + ' AUC' + '=' + auc[i])
    plt.xlim([1e-5, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', facecolor='none', edgecolor='none')
    plt.title(dataset_name + ' without bssr' + ' ROC Curve')
    plt.savefig(img_name)
    plt.show()  # show ROC curve
    return auc

def plot_ROC2(test_labels, resultall, name, dataset_name):
    plt.subplots(num='ROC curve', figsize=[10, 7])
    img_name = dataset_name + ' ROC Curve.png'
    tem = resultall
    auc = []
    for i in range(len(resultall)):
        fpr, tpr, thresholds = metrics.roc_curve(
            test_labels, resultall[i], pos_label=1)  # caculate False alarm rate and Probability of detection
        auc.append("%.5f" % metrics.auc(fpr, tpr))  # caculate AUC (Area Under the Curve)
        print('%s_AUC: %s' % (name[i][0], auc[i]))
        if not i: my_plot = plt.semilogx if metrics.auc(fpr, tpr) > 0.9 else plt.plot
        my_plot(fpr, tpr, label=name[i] + ' AUC' + '=' + auc[i])
    plt.xlim([1e-5, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', facecolor='none', edgecolor='none')
    plt.title(dataset_name + ' without bssr' + ' ROC Curve')
    plt.savefig(img_name)
    plt.show()  # show ROC curve
    return auc


def get_target_index(grt):
    index = []
    for i in range(grt.shape[0]):
        for j in range(grt.shape[1]):
            if grt[i][j] == 1:
                index.append([i, j])
    return index


def get_target_spectral(grt, img):
    index = get_target_index(grt)
    target_spectral = img[..., index[0][0], index[0][1]].reshape(-1, 1)
    for i in range(1, len(index)):
        target_spectral = np.append(target_spectral, img[..., index[i][0], index[i][1]].reshape(-1, 1),
                                    axis=1)
    return target_spectral


def show_spectral(spectral, color, title):
    for j in range(len(spectral)):
        for i in range(spectral[j].shape[1]):
            plt.plot(range(1, spectral[j].shape[0] + 1), spectral[j][..., i], color=color[j], linewidth=1)
    plt.title(title)
    plt.legend()
    plt.show()
