import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import GBIS
import loss
import model
import utils
from Data import GetData, ConstructMat
from scipy import io as scio
import torchinfo
import log


# seed
torch.manual_seed(1)

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hsi_img_path = 'Sandiego.mat'

# ita
superpixels_threshold = 1e-3
# Mt
superpixels_mst = 5

sigma = 2e3
lambda_b = 1
lambda_t = 2
train_times = 1
#theta
coarse_judge_threshold_list = [25]

try:
    hsi_data = GetData(hsi_img_path)
except:
    constructData = ConstructMat(hsi_img_path, flatten=False, normalize=True, reshape_row=False)
    constructData.construct()
    hsi_data = GetData(hsi_img_path)

hsi_img = hsi_data.img
target_spectral = hsi_data.tgt
gt = hsi_data.grt
dataset_name = hsi_data.name
pca_num = 50
input_demension = target_spectral.shape[0]
hsi_img_pca = utils.pca(pca_num, hsi_img.reshape(hsi_img.shape[0], hsi_img.shape[1] * hsi_img.shape[2]))
hsi_img_pca = hsi_img_pca.reshape((pca_num, hsi_img.shape[1], hsi_img.shape[2]))
# model_path = './' + dataset_name + '_model/'
background_model_path = 'b_model_'+ dataset_name + '.pth'
target_model_path = 't_model_' + dataset_name + '.pth'

learning_rate_background = 0.01
learning_rate_target = 0.01
batch_size = 32
background_epoch_total = 4000
target_epoch_total = 4000

background_train_patch_num = 4000
background_test_patch_num = 4000
target_train_patch_num = 4000
target_test_patch_num = 4000

target_batch = np.copy(target_spectral).reshape(-1, 1)
# for i in range(1, batch_size):
#     target_batch = np.append(target_batch, np.copy(target_spectral).reshape(-1, 1), axis=1)
target_batch_tensor = torch.from_numpy(target_batch)
target_batch_tensor = target_batch_tensor.to(device)
target_batch_tensor = target_batch_tensor.type(torch.cuda.FloatTensor)

log_flag = False

background_train_loss = []
background_test_loss = []
target_train_loss = []
target_test_loss = []

def train(model, data_loader, optim, flag, phase='training'):  # m_ae,
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
    running_loss = 0.0
    for index, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device)
        data = data.type(torch.cuda.FloatTensor)
        label = label.type(torch.cuda.FloatTensor)
        if phase == 'training':
            optim.zero_grad()
        output = model(data)
        loss = SDLoss(output, label, target_batch_tensor, flag)
        running_loss += SDLoss_pred(output, label, target_batch_tensor, flag).item()
        if phase == 'training':
            loss.backward()
            optim.step()
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss


def detection(b_model, t_model, test_data_loader, sigma, lambda_b, lambda_t):
    b_model.eval()
    t_model.eval()

    background_result = np.zeros((hsi_img.shape[1], hsi_img.shape[2]))
    target_result = np.zeros((hsi_img.shape[1], hsi_img.shape[2]))
    for index, (test_data, target_data, label) in enumerate(test_data_loader):
        test_data = test_data.to(device)
        test_data = test_data.type(torch.cuda.FloatTensor)
        label = label.detach().numpy()
        target_data = target_data.to(device)
        target_data = target_data.type(torch.cuda.FloatTensor)
        background_output_tensor = b_model(test_data)
        target_output_tensor = t_model(test_data)
        background_output = background_output_tensor.cpu().detach().numpy()
        target_output = target_output_tensor.cpu().detach().numpy()

        '''standard output'''
        background_result_f2_batch = np.linalg.norm(background_output - label, ord=2, axis=1)
        target_result_f2_batch = np.linalg.norm(target_output - label, ord=2, axis=1)

        background_result[index, ...] = background_result_f2_batch # + background_result_sam_batch
        target_result[index, ...] = target_result_f2_batch # + target_result_sam_batch

    '''standard result'''
    row_result = np.power(sigma, utils.background_control(background_result, lambda_b) - utils.background_control(target_result, lambda_t))
    result = utils.mean_anomalous_values(row_result, 1)

    return result


# train(10,bs_model, ba_model,background_train_dataloader,10, batch_size)
# print(1)
#


for i in range(train_times):
    coarse_judge_threshold = coarse_judge_threshold_list[i]

    b_model = model.sae_background((int((target_spectral.shape[0] - 21) / 4) - 34) * 20, target_spectral.shape[0])
    b_model = b_model.to(device)
    t_model = model.sae_target((int((target_spectral.shape[0] - 11) / 2) - 36) * 10, target_spectral.shape[0])
    t_model = t_model.to(device)
    SDLoss = loss.SDLoss()
    SDLoss_pred = loss.SDLoss(size_average=False)
    SDLoss.to(device)
    SDLoss_pred.to(device)

    background_optim = torch.optim.SGD(b_model.parameters(), lr=learning_rate_background, momentum=0.9)
    target_optim = torch.optim.SGD(t_model.parameters(), lr=learning_rate_target, momentum=0.9)

    # background_model_path = model_path + 'b_model_' + dataset_name + '_coarse_judge_threshold=' + str(coarse_judge_threshold)+'_train_times=' + str(i) + '.pth'
    # target_model_path = model_path + 't_model_' + dataset_name + '_coarse_judge_threshold=' + str(coarse_judge_threshold)+'_train_times=' + str(i) + '.pth'
    seg_flag = True
    try:
        b_state_dict = torch.load(background_model_path)
        b_model.load_state_dict(b_state_dict)
    except:
        if seg_flag:
            seg_flag = False
            log_flag = True
            # print(b_model)
            # torchinfo.summary(b_model, (batch_size, 1, target_spectral.shape[0], 3, 3))
            print("{}Start Segment Superpiexls{}\n".format('-' * 25,
                                                                '-' * 25))
            idxNum, imgIdx = GBIS.SegmentImage(hsi_img, superpixels_threshold, superpixels_mst)
            plt.figure(3)
            plt.imshow(imgIdx)
            plt.show()
            print("{}Start Construct Training Patch{}\n".format('-' * 25,
                                                                '-' * 25))
            background_patch, target_patch, background_center_spectral, target_center_spectral, \
            background_patch_test, target_patch_test, background_center_spectral_test, target_center_spectral_test \
                = utils.construct_patch(imgIdx, idxNum, hsi_img, hsi_img_pca, target_spectral, coarse_judge_threshold, target_train_patch_num,
                                        target_test_patch_num,background_train_patch_num, background_test_patch_num)

            target_train = utils.get_dataset(target_patch, target_center_spectral)
            background_train = utils.get_dataset(background_patch, background_center_spectral)
            target_test = utils.get_dataset(target_patch_test, target_center_spectral_test)
            background_test = utils.get_dataset(background_patch_test, background_center_spectral_test)

            target_train_dataloader = DataLoader(target_train, batch_size, shuffle=True, drop_last=True)
            background_train_dataloader = DataLoader(background_train, batch_size, shuffle=True, drop_last=True)
            target_test_dataloader = DataLoader(target_test, batch_size, drop_last=True)
            background_test_dataloader = DataLoader(background_test, batch_size, drop_last=True)

        print("{}Start Training Background Model{}\n".format('-' * 25,
                                                       '-' * 25))
        pbar_b = tqdm(range(0, background_epoch_total))
        flag = 'background'
        smallest_loss = 9999
        for epoch, element in enumerate(pbar_b):
            train_epoch_loss = train(b_model, background_train_dataloader, background_optim, 'b')  # ba_model,
            test_epoch_loss = train(b_model, background_test_dataloader, background_optim, 'b', phase='validation')  # , ba_model
            pbar_b.set_description(f"Epoch {epoch}/{background_epoch_total}")
            pbar_b.set_postfix({"class": flag}, train_loss=train_epoch_loss, test_loss=test_epoch_loss)
            background_train_loss.append(train_epoch_loss)
            background_test_loss.append(test_epoch_loss)
            if test_epoch_loss < smallest_loss:
                smallest_loss = test_epoch_loss
                torch.save(b_model.state_dict(), background_model_path)
                pbar_b.set_postfix({"class": flag}, {"model saved epoch": epoch}, train_loss=train_epoch_loss, test_loss=test_epoch_loss)
        plt.figure(0)
        plt.semilogy(background_train_loss, 'b', label='train loss')
        plt.semilogy(background_test_loss, 'r', label='test loss')
        plt.title("background train loss")
        plt.legend()
        plt.show()
    try:
        t_state_dict = torch.load(target_model_path)
        t_model.load_state_dict(t_state_dict)
    except:
        if seg_flag:
            seg_flag = False
            log_flag = True
            # torchinfo.summary(t_model, (batch_size, 1, target_spectral.shape[0], 3, 3))
            print("{}Start Segment Superpiexls{}\n".format('-' * 25,
                                                           '-' * 25))
            idxNum, imgIdx = GBIS.SegmentImage(hsi_img, superpixels_threshold, superpixels_mst)
            # plt.figure(3)
            # plt.imshow(imgIdx)
            # plt.show()
            print("{}Start Construct Training Patch{}\n".format('-' * 25,
                                                                '-' * 25))
            background_patch, target_patch, background_center_spectral, target_center_spectral, \
            background_patch_test, target_patch_test, background_center_spectral_test, target_center_spectral_test \
                = utils.construct_patch(imgIdx, idxNum, hsi_img, hsi_img_pca, target_spectral, coarse_judge_threshold,
                                        target_train_patch_num,
                                        target_test_patch_num, background_train_patch_num, background_test_patch_num)

            target_train = utils.get_dataset(target_patch, target_center_spectral)
            background_train = utils.get_dataset(background_patch, background_center_spectral)
            target_test = utils.get_dataset(target_patch_test, target_center_spectral_test)
            background_test = utils.get_dataset(background_patch_test, background_center_spectral_test)

            target_train_dataloader = DataLoader(target_train, batch_size, shuffle=True, drop_last=True)
            background_train_dataloader = DataLoader(background_train, batch_size, shuffle=True, drop_last=True)
            target_test_dataloader = DataLoader(target_test, batch_size, drop_last=True)
            background_test_dataloader = DataLoader(background_test, batch_size, drop_last=True)

        print("{}Start Training Target Model{}\n".format('-' * 25,
                                                         '-' * 25))
        pbar_t = tqdm(range(0, target_epoch_total))
        flag = 'target'
        smallest_loss = 9999
        for epoch, element in enumerate(pbar_t):
            train_epoch_loss = train(t_model, target_train_dataloader, target_optim, 't')  # ba_model,
            test_epoch_loss = train(t_model, target_test_dataloader, target_optim, 't', phase='validation')  # , ba_model
            pbar_t.set_description(f"Epoch {epoch}/{target_epoch_total}")
            pbar_t.set_postfix({"class": flag}, train_loss=train_epoch_loss, test_loss=test_epoch_loss)
            target_train_loss.append(train_epoch_loss)
            target_test_loss.append(test_epoch_loss)
            if test_epoch_loss < smallest_loss:
                smallest_loss = test_epoch_loss
                torch.save(t_model.state_dict(), target_model_path)
                pbar_t.set_postfix({"class": flag}, {"model saved epoch": epoch}, train_loss=train_epoch_loss, test_loss=test_epoch_loss)
        plt.figure(1)
        plt.semilogy(target_train_loss, 'b', label='train loss')
        plt.semilogy(target_test_loss, 'r', label='test loss')
        plt.title("target train loss")
        plt.legend()
        plt.show()

    print("{}Start Detection{}\n".format('-' * 25, '-' * 25))

    detection_patch, detection_target, detection_label = utils.construct_test_img(hsi_img, target_spectral)
    detection_patch_data = utils.get_dataset_test(detection_patch, detection_target, detection_label)
    detection_patch_dataloader = DataLoader(detection_patch_data, hsi_img.shape[2])

    detection_result = detection(b_model, t_model, detection_patch_dataloader, sigma, lambda_b, lambda_t)
    scio.savemat('./result_mat/' + dataset_name + '_result.mat',{'results': detection_result})
    detact = utils.Detector(hsi_data)
    auc = detact.show([detection_result.T.reshape(-1,1)], ["row"])