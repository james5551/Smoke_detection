# coding: utf-8

# !/usr/bin/python
import urllib
from urllib import request
import requests
import base64
from PIL import Image
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pickle
from ctpn.inference_ctpn.resize_image import resize_image
from utils_all.rotation import rotate
from utils_all.pic_qingxidu import getImageVar
import json
from aip import AipOcr
import time
import dlib
import face_reg.inference_face.main as face_recongnition
import os
import pandas as pd
import sys
import utils_all.jd_sdk as jd_sdk
from io import BytesIO
import copy
from skimage import transform
import json



def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str
