# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2022/4/25 21:13
@Author  : quke
@File    : api.py.py
@Description:
---------------------------------------
'''
from flask import Flask, request
from infer import Inferenve
from flask_restplus import Api, Resource, fields, reqparse

app = Flask(__name__)
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'z-token'
    },
}

api = Api(app, authorizations=authorizations, security='apikey', doc='/')
ns = api.namespace('api', description='nlp调用接口')  # 模块命名空间
classify_obj = Inferenve()

sentence= api.model('sentence', {  # 输入值模型
    'sentence': fields.String(required=True, description="输入字符串进行分类", example='俄达吉斯坦共和国一名区长被枪杀')
})
class DocumentClassify(Resource):
    @ns.expect(sentence)
    def post(self):
        d = request.get_json(silent=True)
        sentences = d.get('sentence')
        d = classify_obj.get_ret(sentences)
        return {
            'data': d,
            'success': 1
        }
ns.add_resource(DocumentClassify, "/data", endpoint="document_classify")  # 文档分类

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

