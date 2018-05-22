# -*- coding: utf-8 -*-

import threading
import os

import requests
from ctypes import WinError
from urllib.error import URLError

exitFlag = 0

root = 'D:/Matlab/DataMining/datasets/image-net'
api = 'http://www.image-net.org/api'


def download(path, url):
    with open(path, 'wb') as fd:
        res = requests.get(url)
        fd.write(res.content)
        fd.close()
        res.close()


class WebThread():
    def __init__(self, tid, path, url):
        # super().__init__()
        self.tid = tid
        self.path = path
        self.url = url

    def run(self):
        print('Starting %d' % self.tid)
        try:
            download(self.path, self.url)
        except UnicodeDecodeError or TimeoutError:
            print('Failed[%d]:%s' % (self.tid, self.url))
        else:
            print('Success[%d]:%s' % (self.tid, self.url))


def ImageNet(api, root, wnid):
    # 创建文件夹
    folder = root + '/Images/%s/' % wnid
    if not os.path.exists(folder):
        os.mkdir(folder)
    # 获取图片urls和bbox
    download(root + '/%s-urls.txt' % wnid, '%s/text/imagenet.synset.geturls?wnid=%s' % (api, wnid))
    download(root + '/%s-bbox.txt' % wnid, '%s/download/imagenet.bbox.synset?wnid=%s' % (api, wnid))
    # 多线程下载图片
    tid = 0
    # threads = []
    with open(root + '/%s-urls.txt' % wnid) as fd:
        urls = fd.readlines()
        for url in urls:
            url = url.strip()
            tid = tid + 1
            file = folder + '/%s_%04d.jpg' % (wnid, tid)
            if os.path.exists(file):
                print('Skip[%04d]:%s' % (tid, url))
            else:
                thread = WebThread(tid, file, url)
                thread.run()
                # threads.append(thread)
        # for thread in threads:
        #     thread.join()
        print("Exiting Main Thread")


ImageNet(api, root, 'n02123394')
