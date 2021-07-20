from flask import Flask, request,render_template, redirect
from flask_restful import Resource, Api
import os
from remove_bg import preprocessing_query
from search import query

app = Flask(__name__)
api = Api(app)

app.config["IMAGE_UPLOADS"] = "static/query"

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == "POST":
        data = request.get_json(force=True)
        validate = preprocessing_query(data['img_path'], data['rect'])
        if not validate:
            return {'error': 'Ban da chon ngoai khu vuc anh. Vui long chon lai'}
        query_path = 'static/after_query/' + data['img_path']
        results = query(query_path)
        return {'results': results}
    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        if request.files:
            print(request.files)
            image = request.files["image"]
            print(image.filename)
            image.save(os.path.join(
                app.config["IMAGE_UPLOADS"], image.filename))
            return {"filename": image.filename}
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)


# class HelloWorld(Resource):
#     def get(self):
#         return {
#             'about': 'Hello World!'
#         }

#     def post(self):
#         some_json = request.get_json()
#         return {'you sent': some_json}, 201


# class Multi(Resource):
#     def get(self, num):
#         return {'result': num * 10}


# class Segment(Resource):
#     def post(self):
#         data = request.get_json()
#         print(data)
#         # return {'img' : remove_bg(data.img_path, data.seed)}


# @app.route('/segment', methods=['GET', 'POST'])
# def segment():
#     if request.method == "POST":
#         data = request.get_json(force=True)
#         return {'img': remove_bg(data['img_path'], data['seed'])}
#         # if request.files:
#         # print(request.files)
#         # image = request.files["image"]
#         # image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
#         # return {"filename": image.filename}
#     # return render_template("upload.html")


# # api.add_resource(Segment, '/segment')
# api.add_resource(Multi, '/multi/<int:num>')
