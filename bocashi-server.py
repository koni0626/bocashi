# coding:UTF-8

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, jsonify
from werkzeug import secure_filename
from util.yolo import CstmYolo
from zc import lockfile
from zc.lockfile import LockError
import time
import glob
import os
import uuid
import shutil
import argparse


yolo = CstmYolo("./cfg/car-person.cfg", "./cfg/car-person.weights", "./cfg/car-person.data",
                "./cfg/number-face.cfg", "./cfg/number-face.weights", "./cfg/number-face.data", 1)

app = Flask(__name__)


UPLOAD_FOLDER = './uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PORT = 0
RETRY_TIMES = 0
WAIT_TIME = 0

@app.route('/download/<filename>')
def download(filename):
        
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/delete/<filename>')
def delete_file(filename):
    response = {"status": 0,
                "file_name": filename,
                "error_message": ""}
    
    delete_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(delete_filename):
        try:
            os.remove(delete_filename)
        except:
            response["status"] = 1
            response["error_message"] = "{}の削除に失敗しました".format(filename)
    else:
        response["status"] = 2
        response["error_message"] = "{}がありません".format(filename)
        
    return jsonify(ResultSet=response)


@app.route('/bocashi', methods=['GET', 'POST'])
def bocashi():
    '''
    インターフェース
    [request parameter]
    number-flg: 1を指定すると有効、1以外、または省略時は無効。
    face-flg:   1を指定すると有効、1以外、または省略時は無効。
    [response]
    ファイルのURLを返却する。
    
    '''
    global RETRY_TIMES
    global WAIT_TIME
    
    response = {"status": 0,
                "file_name": "",
                "error_message": ""}

    if request.method == 'POST':
        img_file = request.files['img_file']
        
        # オプション取得
        car_op = 1
        face_op = 1
        
        if "car_op" in request.form:
            car_op = int(request.form['car_op'])
        else:
            car_op = 1
        
        if "face_op" in request.form:
            face_op = int(request.form['face_op'])
        else:
            face_op = 1
            
        if img_file:
            filename = secure_filename(img_file.filename)
            filename = str(uuid.uuid4())+".jpg"
            img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            for i in range(RETRY_TIMES):
                try:
                    lock = lockfile.LockFile('lock')
                    
                    yolo.Blur(os.path.join(app.config['UPLOAD_FOLDER'], filename), 
                              os.path.join(app.config['UPLOAD_FOLDER'], filename),
                              {"car_op": car_op, "face_op": face_op})
                    
                    lock.close()
                    
                    response["status"] = 0
                    response["error_message"] = ""
                    break
                except LockError:
                    response["status"] = 2
                    response["error_message"] = "二つ以上の要求が行われています"
                    time.sleep(WAIT_TIME)
                else:
                    response["status"] = 4
                    response["error_message"] = "予測に失敗しました"
            
            response["file_name"] = filename

        else:
            response["status"] = 1
            response["error_message"] = "許可されていない拡張子です"
    else:
        response["status"] = 3
        response["error_message"] = "許可していないリクエストです"

    return jsonify(ResultSet=response)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help = True)
    parser.add_argument('-p', '--port', help="ポート番号を指定します(デフォルト5000)", default=5000, type=int)
    parser.add_argument('-r', '--retry_times', help="リトライ回数を指定します(デフォルト10回)", default=10, type=int)
    parser.add_argument('-w', '--wait_time', help="待ち時間を指定します(デフォルト1秒)", default=1, type=int)
    args = parser.parse_args()
    PORT = args.port
    RETRY_TIMES = args.retry_times
    WAIT_TIME = args.wait_time
   
    app.run(host="0.0.0.0", debug=False, port=PORT)
