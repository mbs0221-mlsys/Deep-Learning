# -*- coding: utf-8 -*-

import threading
import os
import re

import urllib
import urllib.request
from urllib.error import URLError

from bs4 import BeautifulSoup

exitFlag = 0

PATH = 'D:/Matlab/DataMining/datasets/image-net'
API = 'http://www.image-net.org/api'


def download(path, url):
    with open(path, 'w') as fd:
        conn = urllib.request.urlopen(url)
        data = conn.read()
        fd.write(data.decode('UTF-8'))
        fd.close()


class WebThread:
    def __init__(self, tid, path, url):
        self.tid = tid
        self.path = path
        self.url = url

    def run(self):
        print('Starting %d' % self.tid)
        try:
            download(self.path, self.url)
        except UnicodeDecodeError or TimeoutError or URLError:
            print('Failed[%d]:%s' % (self.tid, self.url))
        else:
            print('Success[%d]:%s' % (self.tid, self.url))


def ImageNet(api, root, wnid):
    # 创建文件夹
    folder = root + '/Images/%s' % wnid
    if not os.path.exists(folder):
        os.mkdir(folder)
    # 获取图片urls和bbox
    download(root + '/%s-urls.txt' % wnid, '%s/text/imagenet.synset.geturls?wnid=%s' % (api, wnid))
    download(root + '/%s-bbox.txt' % wnid, '%s/download/imagenet.bbox.synset?wnid=%s' % (api, wnid))
    # 多线程下载图片
    i = 0
    with open(root + '/%s-urls.txt' % wnid, 'w') as fd:
        urls = fd.readlines()
        for url in urls:
            i = i + 1
            file = folder + '/%s_%04d.jpg' % (wnid, i)
            if os.path.exists(file):
                print('Skip[%04d]:%s' % (i, url))
            else:
                thread = WebThread(i, file, url)
                thread.run()
        print("Exiting Main Thread")


def bboxParser(root, wnid):
    # 创建文件夹
    folder = root + '/Annotation/%s' % wnid
    if not os.path.exists(folder):
        os.mkdir(folder)
    # 解析Bounding Box
    pattern = re.compile('(\d+)(\.xml)')
    files = os.listdir(folder)
    boxes = {}
    for file in files:
        matches = pattern.search(file)
        index = matches.group(1)
        # group = re.match(pattern, file)
        idx = int(index)
        with open(folder + '/' + file) as fd:
            soup = BeautifulSoup(fd, 'html5lib')
            bnbbox = soup.find('bndbox')
            xmin = bnbbox.find('xmin')
            ymin = bnbbox.find('ymin')
            xmax = bnbbox.find('xmax')
            ymax = bnbbox.find('ymax')
            bbox = [xmin, ymin, xmax, ymax]
            bbox = list(map(lambda x: int(x.text), bbox))
            boxes[idx] = bbox
    return boxes


def saveBBox(root, wnid, boxes):
    # 创建文件夹
    folder = root + '/Annotation/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    # 写入文件
    with open(folder + wnid + '.txt', 'w') as fd:
        for box in boxes:
            xmin, ymin, xmax, ymax = boxes[box]
            fd.write('%d %d %d %d %d\n' % (box, xmin, ymin, xmax, ymax))


ImageNet(API, PATH, 'n02123478')
# boxes = bboxParser(PATH, 'n02084071')
# print(boxes)

# saveBBox(PATH, 'n02084071', boxes)
