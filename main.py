# @Date : 2024-12-20 21:03  # @Author : yjl
# @Date : 2024-11-19 19:30  # @Author : yjl
from torch.utils.data import Dataset, DataLoader
import main_contrastive_parameter_train as ct
from argparse import Namespace
import torch.nn.functional as F
from utils.logging import set_logger
import copy
import pandas as pd
from ativepolicy.utils import MISSING_LABEL, labeled_indices
from sklearn import preprocessing
from PIL import Image
import torch
import numpy as np
from ativepolicy.pool._uncertainty_sampling_fake_label_9labels import UncertaintySampling
from ativepolicy.pool import RandomSampling
from query_strategies.adversarial_deep_fool import AdversarialDeepFool
from utils.resnet_impl import *
from sklearn.metrics import precision_recall_fscore_support
from utils.resnet_big import SupConResNet
import torch.nn as nn
import random
import os
from torchvision.datasets import MNIST
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.datasets.folder import default_loader
from utils.loadmodel import _load_model_parameters
from utils.loadmodel import _load_torch_model_parameters
from query_strategies.coreset import CoresetSelector

from torchvision.models import resnet34
from torchvision.models import densenet121
from kmeans_sampling import KMeansSampling
import logging
import warnings
warnings.filterwarnings('ignore')
# 多GPU训练--------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 根据cuda是否可用来选择设备，如果cuda可用，设备为cuda0
opt = Namespace  # 创建一个命名空间对象opt,用于存储各种配置选项
"""以下设置模型训练时的超参数和相关选项"""
opt.print_freq = 10  # 打印频率
opt.save_freq = 50  # 保存频率
opt.batch_size = 32  # 批量大小
opt.num_workers = 16  #
opt.learning_rate = 0.005  # 学习率
opt.lr_decay_epochs = [700, 800, 900]
opt.lr_decay_rate = 0.1
opt.weight_decay = 0.0001
opt.momentum = 0.9
opt.dataset = "path"
opt.mean = None
opt.std = None
# opt.data_folder = "./datasets/"
opt.size = 224  # 224
opt.temp = 0.07
opt.cosine = False
opt.syncBN = False
opt.warm = False
opt.trial = 0

opt.epochs = 100  # 100  # contrastive learning epochs 设置对比学习的轮数（epochs）
opt.n_cycles = 70  # active learning running cycles  #100 设置主动学习的周期数
opt.dataset = "path"  # 设置数据集的路径
opt.ct_model_head = "mlp"  # 设置对比学习模型的头部类型
opt.modeltype = "standard"  # standard no_standard  #选择模型类型，可能是standard 或no_standard
opt.savefilename = 'supcon_cycle.pth'  # 设置保存模型的文件名
opt.contrastive_learning_flag = 1  # 设置对比学习的标志   0 表示不执行 对比学习  1：表示执行对比学习
# opt.save_folder_org = r"/opt/data/private/zr/result/test/batch_30_actl_120"
opt.save_folder_org = r"/home/aiusers/space_yjl/lung_cancer/reslut/test/batch_30_actl_120"  # 设置保存结果的文件夹路径
opt.run_name_extension = ""  # 设置运行名称的扩展，可用于标识不同的实验运行
opt.device = device
opt.actl_out_dim = 128  # 设置主动学习任务中的输出维度
opt.actl_num_epochs = 40  # 40
opt.actl_batch_size = 32  # 16
opt.actl_learning_rate = 0.00001  # 0.0001  0.0005 #0.00001  0.005
opt.actl_fixed_init_dataset_flag = False  # True: 从 csv 里面读出       设置是否使用固定的的初始数据集，如果为True,则从CSV文件中读取。
opt.criterion = nn.CrossEntropyLoss()  # 设置损失函数为交叉熵损失函数
def run_save_name(opt):
    # """这段代码主要是将一些实验配置的属性值整合成一个字符串，以形成一个唯一的实验标识符，该标识符可以用来保存结果文件或其他实验相关的操作。"""
    if opt.query_type == "UncertaintySampling":
        queryinfo = "{}8{}".format(opt.query_type, opt.query_method)  # UncertaintySampling8entropy
    elif opt.query_type == "RandomSampling":
        queryinfo = opt.query_type
        # queryinfo = "{}8{}".format(opt.query_type, opt.query_method)
    elif opt.query_type == "CoresetSelector":
        queryinfo = opt.query_type
    elif opt.query_type == "KMeansSampling":
        queryinfo = opt.query_type
    elif opt.query_type == "AdversarialDeepFool":
        queryinfo = opt.query_type
    if opt.model_weight_path == "":
        weight_type = "empty"
    else:
        weight_type = "load"  # 执行这个
    method = opt.method  # SimCLR
    if opt.contrastive_learning_flag == 0:  # 设置对比学习的标志   0 表示不执行 对比学习  1：表示执行对比学习
        method = "NoCtl"

    # method, opt.modeltype, opt.ct_model_head, weight_type, opt.epochs, queryinfo, opt.n_cycles, opt.model_name
    modeltype = opt.modeltype.replace("_", "2")  # standard
    ct_model_head = opt.ct_model_head.replace("_", "2")  # mlp
    queryinfo = queryinfo.replace("_", "2")  # UncertaintySampling8entropy
    model_name = opt.model_name.replace("_", "2")

    run_save_name = "{}_Active_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(opt.method_name,opt.k_folder, method, model_name, modeltype,
                                                                  ct_model_head, weight_type, opt.epochs, queryinfo,
                                                                  opt.n_cycles, opt.run_name_extension)
    return run_save_name

"""主要用于初始化和配置实验中的一些参数和文件路径"""

def opt_init(opt):
    opt.run_save_name = run_save_name(
        opt)  # opt.run_save_name= Active_0_Noctl_arlnet34_standard_mlp_load_100_UncertaintySampling8entropy_70_
    print("run_save_name: {}".format(opt.run_save_name))
    opt.save_folder = os.path.join(opt.save_folder_org, opt.run_save_name)
    opt.data_folder = os.path.join(opt.data_folder_org, opt.run_save_name)

    opt.actl_print_one_flag = False
    opt.model_print_flag_actl = False
    opt.model_print_path_actl = os.path.join(opt.save_folder, "active_Learning_model_parameter.csv")
    opt.model_print_flag_ct = True
    opt.ct_print_one_flag = False
    opt.model_print_path_ct = os.path.join(opt.save_folder, "contrastive_learning_model_parameter.csv")

    if not os.path.exists(opt.actl_init_target_folder):
        os.makedirs(opt.actl_init_target_folder)

    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    if not os.path.exists(opt.data_folder):
        os.makedirs(opt.data_folder)

    if not os.path.exists(opt.pr_excel):
        os.makedirs(opt.pr_excel)

    for classname in opt.classes:
        class_path = os.path.join(opt.data_folder,
                                  classname)  # Active_0_NoCtl_arlnet34_standard_mlp_load_100_UncertaintySampling8entropy_70_ 下有三个类别
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        else:
            shutil.rmtree(class_path)
            os.makedirs(class_path)
    """这一行代码创建了一个名为logger的日志记录器（logger）它接受三个参数："log123"（日志的名称），use_tb_logger=False（不使用 TensorBoard 日志记录器），log_path=opt.save_folder（设置日志文件保存路径为 opt.save_folder）"""
    logger = set_logger("log123", use_tb_logger=False, log_path=opt.save_folder)
    opt.logger = logger
    logger.setLevel(logging.INFO)  # 设置了日志记录器的日志级别为 INFO，这意味着只有 INFO 级别及以上的日志信息会被记录，低于 INFO 级别的日志将被忽略。

    logger.info("run_save_name: {}".format(opt.run_save_name))
    logger.info("data_folder: {}".format(opt.data_folder))
    logger.info("save_folder: {}".format(opt.save_folder))
    # logger.info("batch_size: {}".format(opt.batch_size))
    logger.info("model_name: {}".format(opt.model_name))
    logger.info("method: {}".format(opt.method))
    logger.info("size: {}".format(opt.size))
    logger.info("supcon_epochs: {}".format(opt.epochs))
    logger.info("data: {}".format(opt.dataset))
    logger.info("classes: {}".format(opt.classes))
    logger.info("query_batch: {}".format(opt.query_batch))
    logger.info("active_learning_cycle: {}".format(opt.n_cycles))

    return opt


# ct.trigger(opt)
"""实现了将指定索引的训练数据从源路径复制到目标路径的功能"""
import shutil


def copy_train_data(train_dataset, query_idx, opt):
    train_dataset.idx_to_class = {}
    for key in train_dataset.class_to_idx.keys():
        train_dataset.idx_to_class[train_dataset.class_to_idx[key]] = key
    # src_path = train_dataset.samples[query_idx[0]][0]
    for inx in query_idx:
        src_path = train_dataset.samples[inx][0]
        dest_path = opt.data_folder
        dest_path = os.path.join(dest_path, train_dataset.idx_to_class[train_dataset.samples[inx][1]])
        file_name = os.path.split(src_path)[1]
        dest_path = os.path.join(dest_path, file_name)
        shutil.copy(src_path, dest_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(10)
random_state = np.random.RandomState(0)


class Ative_Classifier():
    def __init__(self, estimator, datasets, test_datasets, num_epochs, batch_size, learning_rate, device, fid,
                 missing_label=MISSING_LABEL, fixed_init_datasets_flag=False):
        # test_dataset.data
        # test_dataset.targets

        self.estimator = estimator
        self.datasets = datasets  # 这个是一个壳子，用于放入 X，y
        self.test_datasets = test_datasets
        self.fid = fid
        # Hyper parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.device = device  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def datasets_preprocess(self, datasets):
        data = datasets.data
        targets = datasets.targets

        self.target_transform = preprocessing.LabelEncoder()
        self.targets_transform = self.target_transform.fit_transform(targets)
        self.data = data
        self.classes = list(np.unique(self.targets_transform))
        return datasets

    def predict(self):
        pass

    # def predict_proba(self,X):
    #     y_predict = self.clf.predict_proba(X)
    #     return y_predict
    #  y_predict.shape  (441, 3)  [[0.58266596, 0.15200112, 0.26533292]]

    def predict_proba(self, X):
        import gc
        output = torch.tensor([])  # 创建一个空的 PyTorch 张量，用于存储所有批次的概率预测结果。
        estimator_cpu = self.estimator.to("cpu")
        for i in range(0, X.shape[0], self.batch_size):  ##数据以小批次为步长循环总数据
            batch = X[i:i + self.batch_size]  # 从输入数据中选择一个大小为self.batch_size的子集，作为当前处理的批次（batch）。即将输入数据分成多个小批次（batch）
            y_predict = estimator_cpu(batch)  # 估计器（estimator）是一种用于拟合数据并进行预测的模型，这里为每个批次的数据进行类别标签（应该得到0-9的值）
            gc.collect()
            p = torch.nn.functional.softmax(y_predict, dim=1)
            if output.shape[0] == 0:
                output = p
            else:
                output = torch.cat([output, p])

            del batch
            del y_predict
            del p
            torch.cuda.empty_cache()
        output = output.cpu().detach().numpy()
        return output

    import copy
    ## filtering np.nan data out
    def _datasets_filtering(self, datasets):
        y = datasets.targets_full
        X = datasets.data_full

        idx_list = np.argwhere(np.isnan(y))
        idx_list = np.squeeze(idx_list).tolist()
        dim = y.shape[0]
        set1 = set(np.arange(dim))
        set2 = set(idx_list)
        data_idx_list = list(set1 - set2)
        """返回一个未标记的数据集"""
        unlabeldata_idx_List = list(set1.intersection(set2))

        X_unlabel = X[unlabeldata_idx_List, :]

        X_left = X[data_idx_list, :]
        y_left = y[data_idx_list]

        datasets.data = X_left
        datasets.targets = y_left
        return datasets,X_unlabel,unlabeldata_idx_List

    # def fit(self, datasets):
    def fit(self, c, result_folder, criterion, n_cycles, best_accuracy, opt):
        from sklearn.metrics import confusion_matrix
        flower_list = self.datasets.class_to_idx_en
        cla_dict = dict((val, key) for key, val in flower_list.items())

        datasets,X_unlabel,unlabeldata_idx_List = self._datasets_filtering(self.datasets)

        # datasets.targets.tolist().count(0)

        print("data size: {}".format(len(datasets)))
        # a=datasets[0]
        from torch.utils.data import DataLoader
        train_loader = torch.utils.data.DataLoader(dataset=datasets,  # datasets=3
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        """train_loader=1,test_loader=7"""
        test_loader = torch.utils.data.DataLoader(dataset=self.test_datasets,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.estimator.parameters(), lr=self.learning_rate)

        # Train the model
        total_step = len(train_loader)
        self.estimator = self.estimator.to(self.device)
        for epoch in range(
                self.num_epochs):  # 在每个 epoch 中，你通过内部循环迭代访问整个训练数据集，逐个处理小批次。在每个小批次中，你执行模型的前向传播（计算预测值）、计算损失、进行反向传播（计算梯度）和优化参数（使用梯度下降算法）的步骤。执行完所有批次后，一个 epoch 完成。外部循环继续，开始下一个 epoch。
            # print("epoch: {}".format(epoch))
            for i, (images, labels) in enumerate(train_loader):
                # print(i)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Forward pass
                outputs = self.estimator(images)

                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("training")
                if (i + 1) % 5 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.num_epochs, i + 1, total_step, loss.item()))

                del images
                del labels
                torch.cuda.empty_cache()

        self.estimator.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)



        y_true = []
        y_pred = []

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.estimator(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
            # 计算混淆矩阵
            print(confusion_matrix(y_true, y_pred))

            self.fid.write("{}\n".format(str(confusion_matrix(y_true, y_pred))))
            current_accuracy = 100 * correct / total
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                torch.save(self.estimator.state_dict(), save_path)

            print(f'Epoch {c + 1}/{n_cycles},Current Accuracy: {current_accuracy} %,Best Accuracy:{best_accuracy}%')


            performance_list = precision_recall_fscore_support(y_true, y_pred)

            weight_performance_list = precision_recall_fscore_support(y_true,
                                                                      y_pred,
                                                                      average='weighted')

        def performance_output(performance_list, weight_performance_list, accuracy, cla_dict):
            performance_dict = {}
            index = 0
            for items in cla_dict.items():
                performance_dict[items[1]] = [performance_list[0].tolist()[index], performance_list[1].tolist()[index],
                                              performance_list[2].tolist()[index], accuracy,
                                              performance_list[3].tolist()[index],
                                              datasets.targets.tolist().count(items[0]),
                                              datasets.targets_true.tolist().count(items[0])]
                index = index + 1
            performance_dict["weighted"] = [weight_performance_list[0], weight_performance_list[1],
                                            weight_performance_list[2], accuracy, sum(performance_list[3].tolist()),
                                            len(datasets.targets.tolist()), len(datasets.targets_full.tolist())]
            return performance_dict

        performance_dict = performance_output(performance_list, weight_performance_list, current_accuracy, cla_dict)

        return performance_dict, datasets,current_accuracy,best_accuracy,X_unlabel, unlabeldata_idx_List,train_loader


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(100352, num_classes)  # 27*7*32  1568

    def forward(self, x):

        try:
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)  # 将特征图向量化，拉长现在应该是二维的，因为我们要得到的是三分类的结果，要是矩阵的形式
            out = self.fc(out)
        except:
            pass
            print("cnn computation wrong!")
        return out


## init active learning data, at beginning we make sure each label has one sample
def gene_init_ative_data(datasets, init_target_folder=None, fixed_init_dataset_flag=False):
    # classes = datasets.targets_true.unique().tolist()
    classes = np.unique(datasets.targets_true)
    classes = np.unique(datasets.targets_true).tolist()

    if fixed_init_dataset_flag == False:
        dict_index_list = {}
        for class_name in classes:
            index = np.argwhere(datasets.targets_true == class_name)[0].tolist()[0]
            dict_index_list[class_name] = [index]
            datasets.targets_full[index] = datasets.targets_true[index]
        df_index = pd.DataFrame(dict_index_list)
        df_index.to_csv(os.path.join(init_target_folder, "init_target.csv"), index=False)
    else:
        df_index = pd.read_csv(os.path.join(init_target_folder, "init_target.csv"))
        index_list = df_index.iloc[0].to_list()
        for index in index_list:
            datasets.targets_full[index] = datasets.targets_true[index]
    return datasets


def init_active_datasets(datasets, init_target_folder=None, fixed_init_dataset_flag=False):
    ##  存储真实数据
    datasets.data_true = copy.deepcopy(datasets.data)
    datasets.targets_true = copy.deepcopy(datasets.targets)

    ## 生成用于训练的数据，开始保证每个标签有一个数据
    datasets.targets_full = np.full(shape=datasets.targets.shape, fill_value=MISSING_LABEL)
    datasets.data_full = copy.deepcopy(datasets.data_true)
    datasets = gene_init_ative_data(datasets, init_target_folder=init_target_folder,
                                    fixed_init_dataset_flag=fixed_init_dataset_flag)

    # np.argwhere
    ## 保存用于训练的数据
    datasets.targets_full = torch.tensor(datasets.targets_full)
    datasets.targets = copy.deepcopy(datasets.targets_full)
    datasets.data = copy.deepcopy(datasets.data_full)

    return datasets


def update_dataset(dataset_target, dataset_src, shape):
    data = torch.tensor([])
    target_list = []
    for img_path, target in dataset_src.samples:
        # print(img_path)
        # print(target)
        sample = default_loader(img_path)
        # sample = dataset_src.transform(sample)
        if dataset_src.transform is not None:
            sample = dataset_src.transform(sample)
        if dataset_src.target_transform is not None:
            target = dataset_src.target_transform(target)

        if data.shape[0] == 0:
            data = sample
        else:
            data = torch.cat((data, sample), 0)
        target_list.append(target)
    data = data.reshape(-1, *shape)
    dataset_target.data = data
    dataset_target.targets = torch.tensor(target_list)
    dataset_target.classes = dataset_src.classes
    dataset_target.class_to_idx_en = dataset_src.class_to_idx
    dataset_target.samples = dataset_src.samples
    return dataset_target


def load_model_weight(model, model_weight_path, device):
    checkpoint = torch.load(model_weight_path, map_location=device)
    state_dict = checkpoint['state_dict']

    del_key = []
    for key, _ in state_dict.items():
        if "fc" in key:
            del_key.append(key)

    for key in del_key:
        del state_dict[key]

    log = model.load_state_dict(state_dict, strict=False)

    return model




"""在训练模型时使用相同的随机种子，从而使得实验结果可以重现。"""


def seed_torch(seed=1029):  # 随机数种子1029
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import torch
from torch import nn



def generate_model(model_name, model_weight_path, device, num_classes, modeltype, model_print_flag=False,
                   model_print_path=None):
    if model_name == "sa_resnet50":
        #         from utils.self_attention_model import arlnet34
        from utils.self_attention_model import sa_resnet50
        estimator = sa_resnet50()
        if model_weight_path != "":
            if modeltype == "standard":
                estimator = _load_torch_model_parameters(model_weight_path, device, estimator,
                                                         model_print_flag=model_print_flag,
                                                         model_print_path=model_print_path)
            else:
                estimator = _load_model_parameters(model_weight_path, device, estimator,
                                                   model_print_flag=model_print_flag, model_print_path=model_print_path)
            #             dim_in = estimator.fc_.in_features
            dim_in = estimator.fc.in_features
            #             estimator.fc_ = nn.Linear(dim_in, num_classes)
            estimator.fc = nn.Linear(dim_in, num_classes)
        estimator = estimator.to(device)

    elif model_name == 'resnet50':
        from torchvision.models import resnet50
        estimator = resnet50()
        if model_weight_path != "":
            if modeltype == "standard":
                estimator = _load_torch_model_parameters(model_weight_path, device, estimator,
                                                         model_print_flag=model_print_flag,
                                                         model_print_path=model_print_path)
            else:
                estimator = _load_model_parameters(model_weight_path, device, estimator,
                                                   model_print_flag=model_print_flag, model_print_path=model_print_path)
            dim_in = estimator.fc.in_features
            estimator.fc = nn.Linear(dim_in, num_classes)

        estimator = estimator.to(device)


    elif model_name == "resnet34":
        estimator = resnet34()
        if model_weight_path != "":
            if modeltype == "standard":
                estimator = _load_torch_model_parameters(model_weight_path, device, estimator,
                                                         model_print_flag=model_print_flag,
                                                         model_print_path=model_print_path)
            else:
                estimator = _load_model_parameters(model_weight_path, device, estimator,
                                                   model_print_flag=model_print_flag, model_print_path=model_print_path)
        dim_in = estimator.fc.in_features
        estimator.fc = nn.Linear(dim_in, num_classes)


        estimator = estimator.to(device)


    elif model_name == "densenet121":
        estimator = densenet121()

        if model_weight_path != "":
            if modeltype == "standard":
                estimator = _load_torch_model_parameters(model_weight_path, device, estimator,
                                                         model_print_flag=model_print_flag,
                                                         model_print_path=model_print_path, model_name=opt.model_name)
            else:
                estimator = _load_model_parameters(model_weight_path, device, estimator,
                                                   model_print_flag=model_print_flag, model_print_path=model_print_path,
                                                   model_name=opt.model_name)
        in_features = estimator.classifier.in_features
        estimator.classifier = nn.Linear(in_features, num_classes)



    elif model_name == "mobilenetv2":
        from torchvision.models import mobilenet_v2
        estimator = mobilenet_v2()
        if model_weight_path != "":
            if modeltype == "standard":
                estimator = _load_torch_model_parameters(model_weight_path, device, estimator,
                                                         model_print_flag=model_print_flag,
                                                         model_print_path=model_print_path, model_name=opt.model_name)
            else:
                estimator = _load_model_parameters(model_weight_path, device, estimator,
                                                   model_print_flag=model_print_flag, model_print_path=model_print_path,
                                                   model_name=opt.model_name)
        in_features = estimator.classifier[1].in_features
        estimator.classifier[1] = nn.Linear(in_features, num_classes)




    elif model_name == "DenseNet":
        from utils.densenet121_se import DenseNet
        estimator = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16))
        if model_weight_path != "":
            if modeltype == "standard":
                estimator = _load_torch_model_parameters(model_weight_path, device, estimator,
                                                         model_print_flag=model_print_flag,
                                                         model_print_path=model_print_path, model_name=opt.model_name)
            else:
                estimator = _load_model_parameters(model_weight_path, device, estimator,
                                                   model_print_flag=model_print_flag, model_print_path=model_print_path,
                                                   model_name=opt.model_name)
        in_features = estimator.classifier.in_features
        estimator.classifier = nn.Linear(in_features, num_classes)

        estimator = estimator.to(device)

    elif model_name == "efficientnet_v2":
        from torchvision.models import efficientnet_v2_s
        estimator = efficientnet_v2_s(pretrained=False)
        if model_weight_path != "":
            if modeltype == "standard":
                estimator = _load_torch_model_parameters(model_weight_path, device, estimator,
                                                         model_print_flag=model_print_flag,
                                                         model_print_path=model_print_path)
            else:
                estimator = _load_model_parameters(model_weight_path, device, estimator,
                                                   model_print_flag=model_print_flag, model_print_path=model_print_path)
        # in_features = estimator.classifier.in_features
        # estimator.classifier = nn.Linear(in_features, num_classes)
        lastconv_output_channels = list(estimator.classifier._modules.items())[-1][1].in_features
        dropout = 0.5
        estimator.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

    elif model_name == "vit_b_16":
        from torchvision.models import vit_b_16
        print("generate_model: vit_b_16")
        # estimator = vit_b_16(pretrained=False)
        estimator = vit_b_16()
        if model_weight_path != "":
            if modeltype == "standard":
                estimator = _load_torch_model_parameters(model_weight_path, device, estimator,
                                                         model_print_flag=model_print_flag,
                                                         model_print_path=model_print_path)
            else:
                estimator = _load_model_parameters(model_weight_path, device, estimator,
                                                   model_print_flag=model_print_flag, model_print_path=model_print_path)
        in_features = estimator.heads.head.in_features
        estimator.heads.head = nn.Linear(in_features, num_classes)

    return estimator
class Net:
    def __init__(self,device,estimator):
        self.device = device
        self.estimator = estimator
    def get_embeddings_layer(self):#用于获取模型中用于提取特征的层
        self.estimator.eval()
        layers = list(self.estimator.children())[:-2]
        backbone = nn.Sequential(*layers)
        return  backbone

    def get_embeddings(self,data,unlabeldata_idx_List):#用于从给定的数据中提取特征。
        backbone = self.get_embeddings_layer()
        backbone.eval()

        # embeddings = torch.zeros([len(data), 512,7,7])
        feature_list = []
        for i in range(0,len(data),16):
            batch_data = data[i:i+16]
            # batch_inx = unlabeldata_idx_List[i:i+16]
            batch_data = batch_data.to(self.device)
            out = backbone(batch_data)
            feature_list.append(out)
        embeddings = torch.cat(feature_list,dim=0)
        return embeddings

def trigger(opt):
    seed_torch()
    MISSING_LABEL = np.nan
    n_cycles = opt.n_cycles  # 50
    best_accuracy = opt.best_accuracy  # 0
    num_classes = opt.actl_num_classes
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out_dim = opt.actl_out_dim
    # device = "cuda"
    num_epochs = opt.actl_num_epochs  # 40
    batch_size = opt.actl_batch_size  # 16
    learning_rate = opt.actl_learning_rate  # 0.00001# 0.0001  0.0005 #0.00001  0.005
    fixed_init_dataset_flag = opt.actl_fixed_init_dataset_flag  # True: 从 csv 里面读出
    init_target_folder = opt.actl_init_target_folder

    train_path = opt.actl_train_path  # r"/home/users/database/database/medical_images/甲状腺/train_data/set0/train"
    test_path = opt.actl_test_path  # r"/home/users/database/database/medical_images/甲状腺/train_data/set0/val"
    criterion = opt.criterion  # 交叉熵损失函数

    if opt.query_type == "UncertaintySampling":
        qs = UncertaintySampling(method=opt.query_method)  # 'least_confident'
    elif opt.query_type == "RandomSampling":
        qs = RandomSampling()
    elif opt.query_type == "CoresetSelector":
        queryinfo = opt.query_type
    elif opt.query_type == "KMeansSampling":
        queryinfo = opt.query_type
    elif opt.query_type == "AdaptiveAdversarial":
        queryinfo = opt.query_type
    model_weight_path = opt.model_weight_path
    # result_folder = r"/home/users/results/XimagesCLS/Con_AL/20230909/jiazhuanxian/Resnet34_pretrain_supcon_ep100_cycle_mlp_UncertaintySampling"
    result_folder = opt.save_folder
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    logger = opt.logger
    logger.setLevel(logging.INFO)
    logger.info(model_weight_path)
    logger.info(train_path)
    logger.info(test_path)
    logger.info(result_folder)

    logger.info("CNN_train_num_epochs: {}".format(num_epochs))
    logger.info("num_classes: {}".format(num_classes))
    logger.info("batch_size: {}".format(batch_size))
    logger.info("learning_rate: {}".format(learning_rate))
    logger.info("fixed_init_datasets_flag: {}".format(fixed_init_dataset_flag))
    logger.info("init_target_folder: {}".format(init_target_folder))

    # 获取模型estimator
    estimator = generate_model(opt.model_name, model_weight_path, device, num_classes, opt.modeltype)

    k_list = []
    for k, v in estimator.state_dict().items():  # state_dict() 是一个将模型的所有参数转化为字典形式的方法。它返回一个包含模型权重和偏置等参数的字典。对于神经网络模型，state_dict() 的返回值通常是一个嵌套的字典，其中包含了每个层的权重和偏置。
        k_list.append(k)
    df = {}
    df['model_name'] = k_list
    df = pd.DataFrame(df)
    df.to_csv('model_name.csv')

    opt.model_print_flag_actl = True
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    from torchvision import transforms
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    data_transform = {
        "train": train_transform,
        "val": train_transform}

    ## get data set
    train_dataset = MNIST(
        # root='/opt/data/private/zr/dataset/data',
        root=r'/home/aiusers/space_yjl/lung_cancer/minst_data',
        train=True,
        # transform=transforms.ToTensor(),
        download=True)
    train_dataset.data = train_dataset.data[:5000]
    train_dataset.targets = train_dataset.targets[:5000]
    test_dataset = MNIST(
        # root='/opt/data/private/zr/dataset/data',
        root=r'/home/aiusers/space_yjl/lung_cancer/minst_data',
        train=False,
        # transform=transforms.ToTensor()
    )

    from torchvision import transforms, datasets

    train_dataset_new = datasets.ImageFolder(root=train_path,
                                             transform=data_transform["train"])

    test_dataset_new = datasets.ImageFolder(root=test_path,
                                            transform=data_transform["train"])

    shape = (3, 224, 224)
    # shape= (3,32,32)
    dataset_src = train_dataset_new
    dataset_target = train_dataset
    train_dataset = update_dataset(dataset_target, dataset_src, shape)

    a = train_dataset[0]
    dataset_src = test_dataset_new
    dataset_target = test_dataset
    test_dataset = update_dataset(dataset_target, dataset_src, shape)

    fid = open(os.path.join(result_folder, "result.txt"), "w")
    clf = Ative_Classifier(estimator, train_dataset, test_dataset, num_epochs, batch_size, learning_rate, device, fid)

    train_dataset = init_active_datasets(train_dataset, init_target_folder=init_target_folder,
                                         fixed_init_dataset_flag=fixed_init_dataset_flag)
    df_index = pd.read_csv(os.path.join(init_target_folder, "init_target.csv"))
    index_list = df_index.iloc[0].to_list()
    # copy initial index to training folder
    print(train_dataset.class_to_idx)
    copy_train_data(train_dataset, index_list, opt)

    totalnum_dict = {}

    for i in range(num_classes):
        totalnum_dict[i] = len(np.where(train_dataset.targets_true == i)[
                                   0])  # np.where(train_dataset.targets_true == i)：使用 NumPy 的 where 函数找到 train_dataset.targets_true 中值等于 i 的元素的索引。len(...)：计算找到的索引的数量，即该类别的样本数量。使用类别 i 作为键。
    for class_label, totalnum in totalnum_dict.items():
        print(f"类别 {class_label} 的总数: {totalnum}")

    accuracy_list = []
    datasize_list = []

    performance_dict_cycle = {}
    # train_dataset.class_to_idx_en
    flower_list = train_dataset.class_to_idx_en  # 字典{'bengin cases':0,'malignant cases':1,'normal cases':2}
    cla_dict = dict(
        (val, key) for key, val in flower_list.items())  # {0:'benigin cases',1:'malignant',2:'normal cases'}
    for items in cla_dict.items():
        performance_dict_cycle[
            items[1]] = []  # performance_dict_cycle = {'benign cases':[],'malignant cases':[],'normal cases':[]}
    performance_dict_cycle[
        "weighted"] = []  # performance_dict_cycle = {'benign cases':[],'malignant cases':[],'normal cases':[]}


    expert_annotation_set = []#专家标注数量的列表
    pseudo_labels = {}  # 用于存储样本索引及其对应的伪标签信息
    for c in range(n_cycles):
        X = train_dataset.data_full
        y = train_dataset.targets_full

        performance_dict, datasets,current_accuracy,best_accuracy,X_unlabel, unlabeldata_idx_List,train_loader = clf.fit(c, result_folder, criterion, n_cycles, best_accuracy, opt)
        logger.info("Test accuracy is {}".format(performance_dict["weighted"][3]))

        for items in cla_dict.items():  ##{0:'benigin cases',1:'malignant',2:'normal cases'}
            performance_dict_cycle[items[1]].append(performance_dict[items[1]])
        performance_dict_cycle["weighted"].append(performance_dict["weighted"])

        datasize = datasets.data.shape[0]

        num_0 = datasets.targets.tolist().count(0)
        num_1 = datasets.targets.tolist().count(1)
        datasize_list.append(datasize)

        import csv
        from time import perf_counter

        # 在循环外初始化
        query_times = []  # 每轮 query 选择耗时（秒）
        update_times = []  # 每轮模型更新耗时（秒）
        round_times = []  # 每轮总耗时（秒）
        round_infos = []  # 存储每轮的详细信息（可写 CSV）

        n_cycles = opt.n_cycles if hasattr(opt, "n_cycles") else 9  # 如果你有 n_cycles 参数，用它；否则改为实际轮数

        for round_idx in range(n_cycles):
            logger.info(f"=== Active Learning round {round_idx + 1}/{n_cycles} ===")
            round_start = perf_counter()

            # ---------- query selection timing start ----------
            query_start = perf_counter()

            print(X.shape)
            if opt.query_type == "RandomSampling":
                query_idx = qs.query(X=X, y=y)

            elif opt.query_type == "UncertaintySampling":
                (query_idx, min_indices, y_pred_class1, second_min_indices, y_pred_class2, third_min_indices,
                 y_pred_class3,
                 fourth_min_indices, y_pred_class4, fifth_min_indices, y_pred_class5, sixth_min_indices, y_pred_class6,
                 seventh_min_indices, y_pred_class7, eighth_min_indices, y_pred_class8, ninth_min_indices,
                 y_pred_class9) = qs.query(
                    X=X, y=y, clf=clf, batch_size=opt.query_batch)
                logger.info("qs.method: {}".format(qs.method))

            elif opt.query_type == "CoresetSelector":
                print(f"\033[92mShape of X_unlabel: {X_unlabel.shape}\033[0m")
                train_dataset_shape = torch.stack([data[0] for data in train_dataset])
                print(f"\033[92mShape of train_dataset_shape: {train_dataset_shape.shape}\033[0m")
                qs = CoresetSelector(X_unlabel)
                query_idx = qs.query(2)  # 或者改成 opt.query_batch
                print(f"\033[92mShape of query_idx: {query_idx}\033[0m")

            elif opt.query_type == "AdversarialDeepFool":
                qs = AdversarialDeepFool(X_unlabel, estimator)
                query_idx = qs.query(opt.query_batch, unlabeldata_idx_List, X_unlabel)

            elif opt.query_type == "KMeansSampling":
                backbone = Net(opt.device, estimator)  # backbone 包含 get_embeddings
                qs = KMeansSampling(X_unlabel, unlabeldata_idx_List, backbone)
                query_idx = qs.query(opt.query_batch)

            else:
                # 如果还有其他策略，按需添加
                query_idx = qs.query(X=X, y=y)

            query_end = perf_counter()
            # ---------- query selection timing end ----------

            query_time = query_end - query_start
            query_times.append(query_time)

            logger.info("query_idx: {}".format(query_idx))
            copy_train_data(train_dataset, query_idx, opt)
            logger.info("data copy complete")
            print("data copy complete")

            # ---------- model update / training timing start ----------
            # 把你原来每轮训练/模型更新的代码放在下面这段的注释区域里（或调用已有的训练函数）
            train_start = perf_counter()

            # === START: 将下面这一行替换为你的模型训练 / fine-tune / 更新代码 ===
            # 例如: train_one_round(train_loader, model, optimizer, criterion, device)
            # 或者如果你在这里有 for epoch in ...: training loops, 则完整保留这些代码
            # 目前我在这里放一个占位（请替换）:
            # train_one_round(...)  # <- 把你的训练更新调用放在这里
            # === END: 替换点 ===

            train_end = perf_counter()
            # ---------- model update / training timing end ----------

            update_time = train_end - train_start
            update_times.append(update_time)

            round_end = perf_counter()
            round_total = round_end - round_start
            round_times.append(round_total)

            # 保存单轮详细信息，后面写 CSV / 打印
            round_infos.append({
                "round": round_idx + 1,
                "query_time_s": query_time,
                "update_time_s": update_time,
                "round_total_s": round_total,
                "query_type": opt.query_type
            })

            logger.info(
                f"Round {round_idx + 1} times: query {query_time:.4f}s, update {update_time:.4f}s, total {round_total:.4f}s")

        # ========== 循环结束，汇总统计 ==========
        query_times_arr = np.array(query_times)
        update_times_arr = np.array(update_times)
        round_times_arr = np.array(round_times)

        # query 平均和标准差（使用样本标准差 ddof=1；如果只有 1 轮就用 ddof=0）
        ddof = 1 if len(query_times) > 1 else 0
        query_mean = np.mean(query_times_arr)
        query_std = np.std(query_times_arr, ddof=ddof)

        total_training_minutes = np.sum(round_times_arr) / 60.0  # 所有轮次总时长，单位分钟
        # 如果你只想把“训练（model update）”部分相加：
        total_update_minutes = np.sum(update_times_arr) / 60.0

        # 打印格式化输出（类似 Table 2）
        print("\n=== Active Learning timing summary ===")
        print(f"Query time per round (s): {query_mean:.2f} ± {query_std:.2f}")
        print(f"Total time across {len(round_times)} rounds (min): {total_training_minutes:.2f}")
        print(f"Total model-update time only (min): {total_update_minutes:.2f}")

        # 写 CSV（每轮明细）
        csv_path = "al_timing_per_round.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "query_time_s", "update_time_s", "round_total_s",
                                                   "query_type"])
            writer.writeheader()
            for info in round_infos:
                writer.writerow(info)

        print(f"Per-round timing saved to: {csv_path}")

        logger.info("query_idx: {}".format(query_idx))
        copy_train_data(train_dataset, query_idx, opt)
        logger.info("data copy complete")
        print("data copy complete")


        count = 0
        for root, dirs, files in os.walk(opt.data_folder):
            for _ in files:
                count = count + 1

        if opt.contrastive_learning_flag == 1:
            logger.info("contrastive learning training datasize： {}".format(count))
            print("contrastive learning training datasize： {}".format(count))
            print("contrastive learning start")
            logger.info("contrastive learning start")
            ct.trigger(opt)
            print("contrastive learning complete")
            logger.info("contrastive learning complete")
        else:
            print("contrastive learning delete")
            logger.info("contrastive learning delete")

        print("model load start")
        logger.info("model load start")

        if opt.contrastive_learning_flag == 1:
            new_model_weight_path = os.path.join(opt.save_folder, opt.savefilename)
            modeltype = "no_standard"
            logger.info("model_weight_path: {}".format(new_model_weight_path))
        else:
            logger.info("model_weight_path not change")
            modeltype = opt.modeltype
            new_model_weight_path = opt.model_weight_path
            print("model_weight_path not change")

        model_print_path = os.path.join(opt.save_folder, "model_parameters_{}.csv".format(modeltype))
        logger.info("model_print_path: {}".format(model_print_path))

        if opt.actl_print_one_flag == False and opt.model_print_flag_actl == True:
            model_print_flag = True
            opt.actl_print_one_flag = True
        else:
            model_print_flag = False

        # 加载对比学习权重
        estimator = generate_model(opt.model_name, new_model_weight_path, device, num_classes, modeltype,
                                   model_print_flag=model_print_flag, model_print_path=opt.model_print_path_actl)
        clf.estimator = estimator
        print("model load complete")
        logger.info("model load complete")

        y[query_idx] = train_dataset.targets_true[query_idx].double()
        expert_annotation_set.append(query_idx)

        """ficmatch添加"""
        if opt.fixmatch:
            from lung_cancer.randaugmeant_replace import RandAugmentForCT
            # 获取未标记数据的图片路径
            unlabeled_paths = [train_dataset.samples[idx][0] for idx in unlabeldata_idx_List]

            class UnlabeledDatasetWithTransform(Dataset):
                def __init__(self, image_paths, transform=None):
                    self.image_paths = image_paths  # 图像路径列表
                    self.transform = transform  # 应用的增强

                def __len__(self):
                    return len(self.image_paths)

                def __getitem__(self, idx):
                    img_path = self.image_paths[idx]  # 获取图片路径
                    image = Image.open(img_path).convert('RGB')  # 打开图片并确保是 RGB 格式
                    if self.transform:
                        weak_img, strong_img = self.transform(image)  # 通过增强获取弱、强增强图像
                    else:
                        weak_img, strong_img = image, image
                    return weak_img, strong_img, img_path  # 返回弱增强图像、强增强图像和路径

            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            """123为上面的transform 456为去掉RandAugmentForCT  789为为上面的transform 加上RandAugmentForCT，101112在789的基础上将加上RandAugmentForCT的参数更换n=3,131415在101112的基础上将加上RandAugmentForCT的参数更换n=4效果不好"""
            """从同一数据实例的弱增强版本和强增强版本获得的伪标签预测之间强加一致性的方法"""

            class TransformFixMatch(object):
                def __init__(self, mean, std):
                    # self.to_pil = transforms.ToPILImage()  # 用于将张量转为 PIL 图像
                    self.weak = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        # transforms.Resize((224, 224)),
                        # transforms.RandomCrop(size=224, padding=int(224 * 0.125), padding_mode='reflect'),
                        ##########新增的transform弱变化
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                    ])
                    self.strong = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        # transforms.Resize((224, 224)),
                        # transforms.RandomCrop(size=224, padding=int(224 * 0.125), padding_mode='reflect'),
                        # RandAugmentForCT(n=2, m=5),
                        ##########新增的transform强变化
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.RandomAdjustSharpness(sharpness_factor=2),
                        # RandAugmentForCT(n=3, m=5),
                        RandAugmentForCT(n=5, m=7),


                    ])
                    self.normalize = transforms.Compose([
                        transforms.ToTensor(),  # 将 PIL 图像转换回张量
                        transforms.Normalize(mean=mean, std=std)
                    ])

                def __call__(self, x):
                    # 将四维张量中的每一张图片转换为 PIL 图像，应用增强，然后再转回张量
                    # x_pil = self.to_pil(x)
                    weak = self.weak(x)  # 弱增强
                    strong = self.strong(x)  # 强增强
                    # return weak,strong
                    return self.normalize(weak), self.normalize(strong)  # 归一化

            transform = TransformFixMatch(mean=mean, std=std)
            unlabeled_dataset = UnlabeledDatasetWithTransform(unlabeled_paths, transform=transform)
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=opt.batch_size, shuffle=True)

            initial_threshold = 0.7  # 初始阈值
            final_threshold = 0.9  # 最终目标阈值
            num_epochs = opt.n_cycles  # 总的训练轮数v
            initial_temperature = 1.5  # 初始温度
            final_temperature = 1.0  # 最终温度
            zero_Lu_count = 0

            # estimator = generate_model(opt.model_name, model_weight_path, device, num_classes, opt.modeltype)
            # estimator = estimator.to(device)
            # estimator = clf.estimator.to(device)
            # clf.estimator.to(device)

            class SupConCNN(nn.Module):
                """backbone + projection head"""
                def __init__(self, name='densenet121', head='mlp', dim_in=128, feat_dim=128, lung_cancer=None):
                    super(SupConCNN, self).__init__()
                    if name == 'densenet121':
                        self.Encoder = densenet121()
                        in_features = self.Encoder.classifier.in_features
                        self.Encoder.classifier = nn.Linear(in_features, dim_in)
                        # in_features = self.encoder.fc.in_features
                        # self.encoder.fc = nn.Linear(in_features,dim_in)
                    print("head: {}".format(head))
                    if head == 'linear':
                        self.head = nn.Linear(dim_in, feat_dim)
                    elif head == 'mlp':
                        self.head = nn.Sequential(
                            nn.Linear(dim_in, dim_in),
                            nn.ReLU(inplace=True),
                            nn.Linear(dim_in, feat_dim)
                        )
                    else:
                        raise NotImplementedError(
                            'head not supported: {}'.format(head))

                def forward(self, x):
                    # print(x.shape)
                    feat = self.Encoder(x)
                    # print(feat.shape)
                    # feat = F.normalize(self.head(feat), dim=1)
                    feat = F.normalize(feat, dim=1)
                    # print(feat.shape)
                    return feat
            model = SupConCNN(name=opt.model_name, head=opt.ct_model_head, dim_in=128).to(device)  # densenet121 mlp


            optimizer = torch.optim.Adam(estimator.parameters(), lr=opt.actl_learning_rate)
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.actl_learning_rate)
            for epoch in range(
                    opt.n_cycles):
                temperature_schedule = initial_temperature - (initial_temperature - final_temperature) * (
                            epoch / num_epochs)

                # 动态调整阈值，逐渐逼近最终目标阈值
                threshold = initial_threshold + (final_threshold - initial_threshold) * (epoch / num_epochs)
                for i, ((images, labels), (weak_unlabeled, strong_unlabeled, _)) in enumerate(
                        zip(train_loader, unlabeled_loader)):
                    images = images.to(device)
                    labels = labels.to(device)
                    weak_unlabeled = weak_unlabeled.to(device)
                    strong_unlabeled = strong_unlabeled.to(device)
                    batch_size = images.shape[0]

                    def de_interleave(x, size):
                        s = list(x.shape)
                        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

                    def interleave(x, size):
                        s = list(x.shape)  # 获取输入 x 的形状
                        try:
                            return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
                        except RuntimeError as e:
                            # 当 reshape 失败时，打印错误信息和当前的大小信息
                            print(f"Error during interleave with size {size}: {e}")
                            print(f"x.shape = {x.shape}")
                            print(f"total elements = {x.numel()}")  # 打印元素总数
                            raise e  # 重新抛出错误

                    inputs_concat = torch.cat((images, weak_unlabeled, strong_unlabeled))
                    total_batch_size = inputs_concat.shape[0]  # 得到 cat 后的总 batch size
                    total_elements = inputs_concat.numel()  # 获取输入的总元素数
                    # 确保 total_batch_size 和 size 能整除
                    size = 2  # 默认的 size
                    for new_size in range(1, total_batch_size + 1):
                        if total_batch_size % new_size == 0:
                            size = new_size
                            break
                    inputs = interleave(inputs_concat, size).to(device)

                    # logits = clf.estimator(inputs)
                    # logits = estimator(inputs)
                    logits = model(inputs)
                    logits = de_interleave(logits, size)
                    logits_x = logits[:batch_size]
                    logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                    del logits
                    Lx = nn.functional.cross_entropy(logits_x, labels, reduction='mean')
                    pseudo_label = torch.softmax(logits_u_w.detach() / temperature_schedule* 0.9, dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    """kl散度"""# 为了提高伪标签质量，可以引入一种软标签损失，计算 logits_u_w和 logits_u_s 之间的 KL 散度损失，让模型的弱、强增强之间更一致，这样即便低置信度的样本也能获得微调。
                    probs_u_w = F.softmax(logits_u_w, dim=-1)
                    log_probs_u_s = F.log_softmax(logits_u_s, dim=-1)
                    kl_div_loss = F.kl_div(log_probs_u_s, probs_u_w, reduction='batchmean')
                    mask = max_probs.ge(threshold).float()
                    # Lu = (nn.functional.cross_entropy(logits_u_s, targets_u,reduction='none') * mask).mean()
                    Lu = (nn.functional.cross_entropy(logits_u_s, targets_u,reduction='none') ).mean()
                    # if Lu.item() == 0:
                    #     zero_Lu_count += 1
                    #     if zero_Lu_count >= 1:  # 连续3次为0时降低阈值
                    #         threshold = max(threshold - 0.05, 0.5)  # 阈值至少保持在0.5
                    #         zero_Lu_count = 0  # 重置计数器
                    #         # print(f"连续 Lu 为0，阈值降低至: {threshold}")
                    # else:
                    #     zero_Lu_count = 0  # 若 Lu 非0，重置计数器

                    # loss = Lx + 0.2*Lu+0.5*kl_div_loss
                    loss = Lu+kl_div_loss
                    # print(f"\033[34m注释kl_div_loss: \033[92m{kl_div_loss}\033[0m")
                    # print(f"\033[34m注释Lu loss: \033[93m{Lu}\033[0m")
                    # loss = Lu
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # del images
                    # del labels
                    torch.cuda.empty_cache()
#临时将 mask 从 Lu 损失计算中移除  Lu = nn.functional.cross_entropy(logits_u_s, targets_u, reduction='none').mean()
#减少 temperature_schedule pseudo_label = torch.softmax(logits_u_w.detach() / (temperature_schedule * 0.9), dim=-1) # 或乘以更小的比例
# optimizer = torch.optim.Adam(model.parameters(), lr=opt.actl_learning_rate)
        """fixmatch_上述的添加"""




        """"""
        if opt.pseudo_labels_flag==True:
            # 处理伪标签逻辑的通用函数
            def process_pseudo_labels(indices, preds):
                for idx, pred in zip(indices, preds):
                    idx = int(idx)
                    if idx in pseudo_labels:
                        # 第二次查询该样本，比较伪标签
                        if pseudo_labels[idx] == pred:
                            y[idx] = pred  # 使用伪标签
                        else:
                            y[idx] = train_dataset.targets_true[idx].double()  # 使用人工标注
                    else:
                        # 第一次查询该样本，存储伪标签
                        pseudo_labels[idx] = pred
            if len(expert_annotation_set) >= 10:
                print('伪标签查询开始')
            # 处理所有的伪标签索引和对应的标签
                process_pseudo_labels(min_indices, y_pred_class1)
                process_pseudo_labels(second_min_indices, y_pred_class2)
                process_pseudo_labels(third_min_indices, y_pred_class3)
                process_pseudo_labels(fourth_min_indices, y_pred_class4)
                process_pseudo_labels(fifth_min_indices, y_pred_class5)
                process_pseudo_labels(sixth_min_indices, y_pred_class6)
                # process_pseudo_labels(seventh_min_indices, y_pred_class7)
                # process_pseudo_labels(eighth_min_indices, y_pred_class8)
                # process_pseudo_labels(ninth_min_indices, y_pred_class9)

        train_dataset.targets_full = y
    print(f'输出专家总标注数量:{len(expert_annotation_set)}')
    print()
    print(f"输出y中总共的标签数量:{torch.isfinite(y).sum().item()}")

    df = {}
    for items in cla_dict.items():
        df[items[1]] = pd.DataFrame(performance_dict_cycle[items[1]],
                                    columns=['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Support_test',
                                             'Support_train', 'Support_train_max'])
        df[items[1]].to_excel(os.path.join(result_folder, "{}.xlsx".format(items[1])))

    df["weighted"] = pd.DataFrame(performance_dict_cycle["weighted"],
                                  columns=['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Support_test',
                                           'Support_train', 'Support_train_max'])
    df["weighted"].to_excel(os.path.join(result_folder, "weighted.xlsx"), index=False)

    df_performance = {}
    for item in ['Precision', 'Recall', 'F1 Score', 'Accuracy']:
        df_performance[item] = {key: df[key][item].tolist() for key in df.keys()}

        df_performance[item]['Support_test'] = df["weighted"]['Support_test'].tolist()
        df_performance[item]['Support_train'] = df["weighted"]['Support_train'].tolist()
        df_performance[item]['Support_train_max'] = df["weighted"]['Support_train_max'].tolist()
        df_temp = pd.DataFrame(df_performance[item])
        df_temp.to_excel(os.path.join(result_folder, "{}.xlsx".format(item)), index=False)

    logger.info("****************running complete***************")
    logger.info("*******************************e***************")
    logging.shutdown()
    del qs
    del clf
    fid.close()


if __name__ == "__main__":
    import time

    start_time = time.time()


    ######################111222222222222222222222222222222222222####################################
    opt.query_batch = 15
    opt.k_folder = 1
    opt.method_name = "al-uncertainty-sup"
    opt.batch_size = 32
    opt.fixmatch = False  # 使用fixmatch半监督方法
    opt.actl_fixed_init_dataset_flag = False  # false创建csv文件
    opt.balance_flag = 0  # 在对比学习里面定义了一个类平衡数据 main_constrastive_parameter_train.py
    opt.pseudo_labels_flag = False

    opt.pr_excel = r'/home/aiusers/space_yjl/胎儿耻骨数据集/胎儿检测3data_/save/pr_excel_folder'
    opt.data_folder_org = r'/home/aiusers/space_yjl/胎儿耻骨数据集/胎儿检测3data_/save/data_move'
    opt.save_folder_org = r'/home/aiusers/space_yjl/胎儿耻骨数据集/胎儿检测3data_/save/result'
    opt.actl_init_target_folder = r'/home/aiusers/space_yjl/胎儿耻骨数据集/胎儿检测3data_/save/init_target_folderr'
    opt.actl_train_path = r"/home/aiusers/space_yjl/胎儿耻骨数据集/胎儿检测3data_/data/train"
    opt.actl_test_path = r"/home/aiusers/space_yjl/胎儿耻骨数据集/胎儿检测3data_/data/test"

    # opt.pr_excel = r'/home/aiusers/space_yjl/胎儿耻骨数据集/TSNE_data/al_supcon_uncertainty/pr_excel_folder'
    # opt.data_folder_org = r'/home/aiusers/space_yjl/胎儿耻骨数据集/TSNE_data/al_supcon_uncertainty/data_move_6fake'
    # opt.save_folder_org = r'/home/aiusers/space_yjl/胎儿耻骨数据集/TSNE_data/al_supcon_uncertainty/aaa_demo_6fake'
    # opt.actl_init_target_folder = r'/home/aiusers/space_yjl/胎儿耻骨数据集/TSNE_data/al_supcon_uncertainty/init_target_folderr'
    # opt.actl_train_path = r"/home/aiusers/space_yjl/胎儿耻骨数据集/胎儿检测2data/us_data/final/final_split_2class_82/train"
    # opt.actl_test_path = r"/home/aiusers/space_yjl/胎儿耻骨数据集/胎儿检测2data/us_data/final/final_split_2class_82/test"

    # opt.actl_train_path = r"/home/aiusers/space_yjl/dataset/train"
    # opt.actl_test_path = r"/home/aiusers/space_yjl/dataset/val"

    class_list = []  # 返回的是类别结果-‘normal’，‘bad’,'good'
    for categroy in os.listdir(opt.actl_train_path):
        class_list.append(str(categroy))
    opt.classes = class_list  # opt.classes原先是['yes','no']-->划分了类别目录，生成了三个类['Benigin cases','Malignant cases','Normal cases']
    opt.actl_num_classes = len(opt.classes)
    print(opt.actl_num_classes)
    opt.contrastive_learning_flag = 1  # 设置对比学习的标志   0 表示不执行 对比学习  1：表示执行对比学习
    opt.method = "SupCon"  # "SupCon"   "SimCLR"
    # save_path = r'/home/aiusers/space_yjl/lung_cancer/model_weight_path/save/fixmatch_lungcancer.pth'
    save_path = r'/home/aiusers/space_yjl/胎儿耻骨数据集/TSNE_data/al_supcon_uncertainty/al_uncertainty_sup胎儿.pth'
    opt.model_weight_path = r"/home/aiusers/space_yjl/lung_cancer/model_weight_path/densenet121_weights.pth"
    opt.epochs = 100  # 100  # contrastive learning epochs
    opt.n_cycles = 70  # 主动学习迭代次数70

    opt.query_method = 'entropy'  # entropy, least_confident, margin_sampling
    opt.query_type = "UncertaintySampling"  # AdversarialDeepFool RandomSampling  UncertaintySampling CoresetSelector KMeansSampling
    model_name_list = ['densenet121']  # arlnet34  resnet34 sa_resnet50    resnet50 mobilenetv2  densenet121
    for model_name in model_name_list:
        print(f'start_model:{model_name}')
        opt.best_accuracy = 0
        opt.model_name = model_name
        opt = opt_init(opt)
        trigger(opt)





    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_hours = execution_time / 3600
    print(f"程序执行时间: {execution_time_hours:.2f} 小时")

