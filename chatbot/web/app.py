# coding=utf-8
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))

from flask import Flask, render_template, request, jsonify
import time
import threading
import jieba
from evaluate import *

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import logging

jieba.setLogLevel(logging.INFO)

app = Flask(__name__, static_url_path="/static")


@app.route('/message', methods=['POST'])
def reply():
    # 从请求中获取输入信息
    req_msg = request.form['msg']
    # print(req_msg)
    # 将语句使用结巴分词进行分词
    req_msg = " ".join(jieba.lcut(req_msg))

    res_msg = predict(req_msg)

    res_msg = res_msg.replace('_UNK', '^_^')
    res_msg = res_msg.strip()

    # 如果接受到的内容为空，则给出相应的回复
    if res_msg == ' ':
        res_msg = '我没听清，请再说一遍'

    return jsonify({'text': res_msg})


@app.route("/")
def index():
    return render_template("index.html")


# 启动APP
if (__name__ == "__main__"):
    app.run(host='0.0.0.0', port=8808)
