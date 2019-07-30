# coding=utf8

from __future__ import unicode_literals, absolute_import, print_function

import json
import traceback
from django.db.models import Count, Q
from django.core.management import BaseCommand
import logging
import time
from gm_types.gaia import PROBLEM_FLAG_CHOICES
from libs.es import ESPerform
from libs.bodylib import file_to_ndarray
from libs.bodylib import url_to_ndarray
from libs.facelib import face2vec
import numpy as np
import pandas as pd
import faiss
from django.conf import settings
from libs.tools import Tools
#########新加入部分###########
import argparse
import os
import pickle
import time

import faiss
import numpy as np
from PIL import Image
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from deepcluster import clustering
from deepcluster import models
from deepcluster.util import AverageMeter, Logger, UnifLabelSampler
import requests
import io
####################################
######参数配置##########
parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16'], default='alexnet',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                    help='number of cluster for k-means (default: 10000)')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--reassign', type=float, default=1.,
                    help="""how many epochs of training between two consecutive
                    reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: None)')
parser.add_argument('--checkpoints', type=int, default=25000,
                    help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')
#####################
class Command(BaseCommand):

    def generate_diary_face_vec(self):
        try:
            q = {
                "query":{
                    "bool":{
                        "filter":[
                            {"term":{"is_online":True}},
                            {"term":{"has_before_cover":True}},
                            {"term":{"has_after_cover":True}},
                            {"exists": {"field": "before_cover_url"}}
                        ]
                    }
                },
                "_source":{
                    "include":["id","before_cover_url","after_cover_url"]
                }
            }
            # 术后图向量数据
            # diary_cover_pic_face_vec_fd = open("./diary_cover_pic_face_vec.txt", "w")

            # 术前图向量数据
            diary_before_cover_pic_face_vec_fd = open("./diary_before_cover_pic_face_vec.txt", "w")

            for start_index in range(0,200000,100):
                result_dict = ESPerform.get_search_results(ESPerform.get_cli(), "diary", q, offset=start_index, size=100,doc_type="diary")
                # logging.info("duan add,total_count:%d" % result_dict["total_count"])

                for item in result_dict["hits"]:
                    diary_id = item["_source"]["id"]

                    if "before_cover_url" in item["_source"] and len(item["_source"]["before_cover_url"])>0:
                        before_cover_url = item["_source"]["before_cover_url"] + "-w"
                        after_cover_url = item["_source"]["after_cover_url"] + "-w"

                        # logging.info("after_cover_url:%s" % after_cover_url)
                        logging.info("before_cover_url:%s" % before_cover_url)

                        try:
                            begin_time = float(time.time())

                            # img = url_to_ndarray(after_cover_url)
                            #img = url_to_ndarray(before_cover_url)
                            ####导入一张图片，通过神经网络提取特征，features：4096####
                            global args
                            args = parser.parse_args()

                            # fix random seeds
                            torch.manual_seed(args.seed)
                            torch.cuda.manual_seed_all(args.seed)
                            np.random.seed(args.seed)

                            # CNN
                            if args.verbose:
                                print('Architecture: {}'.format(args.arch))
                            model = models.__dict__[args.arch](sobel=args.sobel)
                            fd = int(model.top_layer.weight.size()[1])
                            model.top_layer = None
                            model.features = torch.nn.DataParallel(model.features)
                            model.cuda()
                            cudnn.benchmark = True

                            # create optimizer
                            optimizer = torch.optim.SGD(
                                filter(lambda x: x.requires_grad, model.parameters()),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=10 ** args.wd,
                            )

                            # define loss function
                            criterion = nn.CrossEntropyLoss().cuda()

                            # optionally resume from a checkpoint
                            if args.resume:
                                if os.path.isfile(args.resume):
                                    print("=> loading checkpoint '{}'".format(args.resume))
                                    checkpoint = torch.load(args.resume)
                                    args.start_epoch = checkpoint['epoch']
                                    # remove top_layer parameters from checkpoint
                                    for key in checkpoint['state_dict']:
                                        if 'top_layer' in key:
                                            del checkpoint['state_dict'][key]
                                    model.load_state_dict(checkpoint['state_dict'])
                                    optimizer.load_state_dict(checkpoint['optimizer'])
                                    print("=> loaded checkpoint '{}' (epoch {})"
                                          .format(args.resume, checkpoint['epoch']))
                                else:
                                    print("=> no checkpoint found at '{}'".format(args.resume))

                            # creating checkpoint repo
                            exp_check = os.path.join(args.exp, 'checkpoints')
                            if not os.path.isdir(exp_check):
                                os.makedirs(exp_check)

                            # creating cluster assignments log
                            cluster_log = Logger(os.path.join(args.exp, 'clusters'))

                            # preprocessing of data
                            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                            tra = [transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize]
                            transformer = transforms.Compose(tra)
                            # load the data
                            end = time.time()
                            result = requests.get(before_cover_url, timeout=30)
                            if result.ok:
                                img = Image.open(io.BytesIO(result.content))
                                img = img.convert('RGB')
                            dataset = transformer(img)  #已经转换成tensor
                            dataset=dataset.numpy()
                            dataset=dataset[np.newaxis,:,:,:]
                            dataset=torch.from_numpy(dataset)
                            #dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
                            if args.verbose: print('Load dataset: {0:.2f} s'.format(time.time() - end))
                            '''dataloader = torch.utils.data.DataLoader(dataset,
                                                                     batch_size=args.batch,
                                                                     num_workers=args.workers,
                                                                     pin_memory=True)'''

                            # clustering algorithm to use
                            #deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

                            # training convnet with DeepCluster
                            end = time.time()

                            # remove head
                            model.top_layer = None
                            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

                            # get the features for the whole dataset
                            features = compute_features(dataset, model, 1)
                            faces = []
                            fes={}
                            fes['feature'] = json.dumps(features)
                            faces.append(fes)
                            #faces = face2vec(img)
                            end_time = float(time.time())
                        #######################################
                            for face in faces:
                                # line = str(diary_id) + "\t" + face["feature"] + "\t" + after_cover_url + "\n"
                                line = str(diary_id) + "\t" + face["feature"] + "\t" + before_cover_url + "\n"

                                # diary_cover_pic_face_vec_fd.write(line)
                                diary_before_cover_pic_face_vec_fd.write(line)

                                logging.info("duan add,diary_id:%s,time cost:%fms" % (str(diary_id),(end_time-begin_time)))

                            def compute_features(dataloader, model, N):
                                if args.verbose:
                                    print('Compute features')
                                batch_time = AverageMeter()
                                end = time.time()
                                model.eval()
                                for input_tensor in dataloader:
                                    # discard the label information in the dataloader
                                    input_var = torch.autograd.Variable(dataloader.cuda(), volatile=True)
                                    aux = model(input_var).data.cpu().numpy()
                                    num=np.array(aux[0,:])
                                    features=num.tolist()
                                return features
                        except:
                            logging.error("catch exception, err_msg:%s" % traceback.format_exc())

            diary_before_cover_pic_face_vec_fd.close()
        except:
            logging.error("catch exception, err_msg:%s" % traceback.format_exc())


    def save_faiss_index(self):
        try:
            diary_cover_pic_face_vec_fd = open("./diary_cover_pic_face_vec.txt","r")

            xb, ids = [], []
            for line in diary_cover_pic_face_vec_fd.readlines():
                line_term_list = line.split("\t")
                diary_id = line_term_list[0]
                face_feature = json.loads(line_term_list[1])
                face_feature_vec = np.array(face_feature)
                xb.append(face_feature_vec)
                ids.append(diary_id)

            xb_np = np.array(xb).astype('float32')
            ids_np = np.array(ids).astype('int')
            index = faiss.IndexHNSWFlat(128, 32)
            index = faiss.IndexIDMap(index)
            index.add_with_ids(xb_np, ids_np)
            faiss.write_index(index, settings.INDEX_PATH)

            diary_cover_pic_face_vec_fd.close()
        except:
            logging.error("catch exception, err_msg:%s" % traceback.format_exc())

    def load_index(self):
        try:
            index = faiss.read_index(settings.INDEX_PATH)
            return index
        except:
            logging.error("catch exception, err_msg:%s" % traceback.format_exc())
            return None

    def find_similar_diary_id_by_url(self,url):
        try:
            test_index = self.load_index()
            img = url_to_ndarray(url)
            faces = face2vec(img)
            for face in faces:
                feature = np.array(json.loads(face["feature"])).astype('float32')
                search = [feature]
                D, I = test_index.search(np.array(search), 5)
                ids = list(I.flat)
                for faceids, ranks in zip(I, D):
                    d = {'faceid': faceids, 'rank': ranks}
                    df = pd.DataFrame(data=d)
                    sort_df = df.sort_values('rank')
                    for row in sort_df.iterrows():
                        faceid = row[1]['faceid']
                        rank = row[1]['rank']

                        logging.info("faceid:%s,randk:%s" % (str(faceid),str(rank)))
        except:
            logging.error("catch exception, err_msg:%s" % traceback.format_exc())


    def handle(self, *args, **options):
        # 生成图片向量
        self.generate_diary_face_vec()
        # 生成索引向量
        # self.save_faiss_index()

        #url = "https://pic.igengmei.com/2019/06/06/1640/96e1e59912f3-w"
        #Tools.find_similar_diary_id_by_url(url)
        print("cache index in local disk")