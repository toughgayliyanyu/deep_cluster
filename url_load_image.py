# coding=utf8

from __future__ import unicode_literals, absolute_import, print_function

import json
import requests
import traceback
import os
import sys
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
from urllib.request import urlretrieve
from django.conf import settings
from libs.tools import Tools



class Command(BaseCommand):

    def load_image_from_diaryurl(self):
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
            #diary_before_cover_pic_face_vec_fd = open("./diary_before_cover_pic_face_vec.txt", "w")
            #术前图片导入文件夹
            _path=os.getcwd()
            new_path=os.path.join(_path,'diary_before_cover_pic')
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            new_path+='/'
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
                        urlretrieve(before_cover_url,new_path+'%s.jpg'%diary_id)

            #diary_before_cover_pic_face_vec_fd.close()
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
        self.load_image_from_diaryurl()

        # 生成索引向量
        # self.save_faiss_index()

        #url = "https://pic.igengmei.com/2019/06/06/1640/96e1e59912f3-w"
        #Tools.find_similar_diary_id_by_url(url)
        print("cache index in local disk")

