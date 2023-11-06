import requests
#阈值法调用
params = {'path': 'http://116.63.203.45:10000/%E6%B8%A9%E5%BA%A630%E2%84%83%20%E6%B9%BF%E5%BA%A698%202022%E5%B9%B410%E6%9C%8814%E6%97%A5.jpg',
          # 'threshold_value':140,
          'img_num':1,
          'method':2}
for i in range(1000):
    r = requests.get("http://127.0.0.1:5000/threshold", params=params)
    print(i)


#对比法调用
# params = {'path':'http://116.63.203.45:10000/%E6%B8%A9%E5%BA%A630%E2%84%83%20%E6%B9%BF%E5%BA%A698%202022%E5%B9%B410%E6%9C%8828%E6%97%A5.jpg',
#           'label': 2,
#           'img_num': 1,
#           'threshold': 20
# }
#
# r = requests.get("http://127.0.0.1:5000/compare", params=params)


# #彩色合成
# params = {'path': 'data/MS600_REF_0003_660nm.jpg',
#           'method': 12,
#           'output': 'output/'
# }