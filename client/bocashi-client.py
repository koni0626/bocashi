# coding:UTF-8

import requests
import json
import sys


if __name__ == '__main__':
    url = 'http://192.168.1.86:5000/bocashi'
    file ={'img_file': open('test.jpg', 'rb')}

    print("画像をサーバーにアップロードします")
    options = {"car_op": 0, "face_op": 0}
    res = requests.post(url, files=file, data=options)
    result_dict = json.loads(res.text)
    status = result_dict["ResultSet"]["status"]
    file_name = result_dict["ResultSet"]["file_name"]
    
   # if res.status_code != 200:
        
    if status != 0:
        print(result_dict["ResultSet"]["error_message"])
        sys.exit(status)
    
    
    print("ぼかした画像をダウンロードします")
    url = 'http://192.168.1.86:5000/download/{}'.format(file_name)
    save_filename = "result.jpg"
    res = requests.get(url)
    img = res.content
    with open(save_filename, "wb") as f:
        f.write(img)
    
    
    print("サーバーから画像を削除します")
    url = 'http://192.168.1.86:5000/delete/{}'.format(file_name)
    res = requests.get(url)
    result_dict = json.loads(res.text)
    status = result_dict["ResultSet"]["status"]
    file_name = result_dict["ResultSet"]["file_name"]
    if status != 0:
        print(result_dict["ResultSet"]["error_message"])
        sys.exit(status)
 
    "エラーのテスト"
    print("サーバーから画像を削除します")
    url = 'http://192.168.1.86:5000/delete/{}'.format(file_name)
    res = requests.get(url)
    result_dict = json.loads(res.text)
    status = result_dict["ResultSet"]["status"]
    file_name = result_dict["ResultSet"]["file_name"]
    if status != 0:
        print(result_dict["ResultSet"]["error_message"])
        sys.exit(status)
 

    

    
