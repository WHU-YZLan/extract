# 发送到response的headers里面（客户端）
import asyncio
import io

import cv2
from flask import Flask,session,url_for,request
from numpy import *
from processing import *
from PIL import Image
import json
import time,os
import re
import urllib.request
import requests
import urllib.parse
import PIL
import color_synthesis
import time
#跨域
# from flask_cors import CORS
JSON_NAME='main.log'
'''
这里可以给SECRET_KEY设置一个随机N位字符串：os.urandom(n)
'''
app = Flask(__name__)
# CORS(app,supports_credentials=True)
def download(file_path, picture_url):
	headers = {
		"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 			(KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE",
		}
	r = requests.get(picture_url, headers=headers)
	with open(file_path, 'wb') as f:
		f.write(r.content)
	f.close()


@app.route('/')
def Hello_world():
	return 'Hello world!'


async def test():
	time.sleep(3)
	print('ok')
	return 'ok'
@app.route('/threshold',methods=['GET'])
def threshold():
	T0 = time.time()
	method=int(request.values.get('method'))


	folder_name='data/'
	os.makedirs(folder_name, exist_ok=True)  # 输出目录

	prefix_url = request.values.get('path')  # 同一类目下的图片url前缀
	n = int(request.values.get('img_num'))  # 该类目下的图片总数

	tmp = prefix_url.split('/')[-1]

	for i in range(1, n + 1):
		file_path = folder_name + tmp
		picture_url = prefix_url
		download(file_path, picture_url)
	T5 = time.time()
	print(T5-T0)
	img_dict={}

	# return img_dict
	img_corrective=process(folder_name)
	mask = mask_maker(img_corrective[0], True)
	T1 = time.time()
	#中文转码
	tmp_chinese=urllib.parse.unquote(tmp.replace('.jpg', ''))


	if method==1:
		threshold_value=int(request.values.get('threshold_value'))
		img_dict['method'] = 'manual input'
	else:
		threshold_value=calculate_threshold(img_corrective[0],mask)
		img_dict['method'] = 'automatic input'
	img_dict['name']=tmp_chinese.replace('.jpg','_',1)+'_'+str(threshold_value)+'_threshold.jpg'
	img_dict['threshold']=threshold_value

	img_dict['path']=request.values.get('path')
	img_dict['img_num']=request.values.get('img_num')
	T2 = time.time()
	rate,img_out=calculate_new(img_corrective[0],mask,threshold_value)
	img_dict['rate(%)']=str(rate)
	# img_dict['content']=img_out.tolist()
	T3 = time.time()
	img_dict['paper_extract_time'] = str(T1 - T0)
	img_dict['threshold_calculate__time'] = str(T2 - T1)
	img_dict['mold_extract_time'] = str(T3 - T2)
	# print(img_dict['paper_extract_time'], img_dict['threshold_calculate__time'], img_dict['mold_extract_time'])
	cv2.imencode('.jpg',img_out)[1].tofile('output/'+img_dict['name'])
	json_data=json.dumps(img_dict,ensure_ascii=False)
	JSON_NAME = str(time.time()) + '_main.log'
	with open(JSON_NAME,'w') as json_file:
		json_file.write(json_data)
	json_file.close()
	return json_data

@app.route('/compare',methods=['GET'])
def compare():
	T0 = time.time()
	folder_name = "data/"
	#下载网络图片
	os.makedirs(folder_name, exist_ok=True)  # 输出目录
	file_list = os.listdir("data")
	if len(file_list) != 0:
		for file in file_list:
			os.remove("data/" + file)
	prefix_url = request.values.get('path')  # 同一类目下的图片url前缀
	n = int(request.values.get('img_num'))  # 该类目下的图片总数

	tmp = prefix_url.split('/')[-1]
	for i in range(1, n + 1):
		file_path = folder_name + tmp
		picture_url = prefix_url
		download(file_path, picture_url)
	# 中文转码
	tmp_chinese = urllib.parse.unquote(tmp.replace('.jpg', ''))





	#处理
	label = int(request.values.get('label'))
	os.makedirs(folder_name, exist_ok=True)  # 输出目录
	img_dict = {}
	img_corrective = process(folder_name)
	T1 = time.time()

	if label == 1:
		if not os.path.exists("base"):
			os.makedirs("base")
		file_list = os.listdir("base")
		if len(file_list) != 0:
			for file in file_list:
				os.remove("base/" + file)
		cv2.imwrite('base/base.jpg',img_corrective[0])
		img_dict['name']='base.jpg'
		img_dict['paper_extract_time'] = str(T1 - T0)
	else:
		base=cv2.imread('base/base.jpg')
		mask = mask_maker(img_corrective[0], False)
		T2 = time.time()
		change=img_corrective[0]
		img_dict['name']='compare'
		threshold=int(request.values.get("threshold"))
		img_dict['increase'], img_dict['net_increase'], _ = \
			calculate_compare(base, change, mask, "output/"+img_dict['name'], threshold)
		T3 = time.time()
		img_dict['name']=tmp_chinese+img_dict['name']+'_'+str(img_dict['increase'])+'.jpg'
		img_dict['paper_extract_time'] = str(T1 - T0)
		img_dict['mask_maker'] = str(T2 - T1)
		img_dict['mold_extract_time'] = str(T3 - T2)
	json_data = json.dumps(img_dict)
	JSON_NAME = str(time.time()) + '_main.log'
	with open(JSON_NAME, 'w') as json_file:
		json_file.write(json_data)
	json_file.close()

	# return json_data

@app.route('/Colorsynthesis',methods=['GET'])
def Colorsynthesis():
	color_synthesis.synthesis(request.values.get('path'), request.values.get('method'), request.values.get('output'))







if __name__ == '__main__':
	app.run(host='0.0.0.0')
	# print(asyncio.run(test()))
	# print("111")

