from distutils.log import debug
from flask import Flask,render_template,url_for,request,session,logging,redirect,flash, Response
from flask import send_file
# from sqlalchemy import create_engine
# from sqlalchemy.orm import scoped_session,sessionmaker
import os
from fastai.vision.all import *
import pickle
from pathlib import Path
import shutil
from PIL import Image
import logging



global_learner_object = 0
UPLOAD_FOLDER = '.'

# engine=create_engine("mysql+pymysql://root:@localhost/users")
# db=scoped_session(sessionmaker(bind=engine))

app = Flask(__name__, static_url_path='/',)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.sqlite3'

file_type_list = [".onnx", ".pb", ".meta", ".tflite", ".lite",
                  ".tfl", ".keras", ".h5", ".hd5", ".hdf5", ".json", ".model", ".mar",
                  ".params", ".param", ".armnn", ".mnn", ".ncnn", ".tnnproto", ".tmfile", ".ms", ".nn",
                  ".uff", ".rknn", ".xmodel", ".paddle", ".pdmodel", ".pdparams", ".dnn", ".cmf", ".mlmodel",
                  ".caffemodel", ".pbtxt", ".prototxt", ".pkl", ".pt", ".pth",
                  ".t7", ".joblib", ".cfg", ".xml", ".zip", ".tar"
                  ]
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/infer')
def infer():
    return render_template('inference.html')

@app.route('/runinference', methods=['POST'])
def runinference():
    if request.method == 'POST':
        weights_file = request.files['weights_upload']
        image_file = request.files['img_upload']
        weights_file.save(weights_file.filename)
        image_file.save("static/tmp/" + image_file.filename)
        with open('file.pkl', 'rb') as file:
            learn = pickle.load(file)
        filename = (weights_file.filename).split(".")
        learn.load(filename[0])
        pred = learn.predict(image_file.filename)
    return render_template('inference.html', content = pred[0], fname = "tmp/" + image_file.filename)


# @app.route('/visualize')
# def visualize():
#     return render_template('visualize.html')
@app.route('/visualize')
def visualize():
    index()
    return render_template('visualize.html')


@app.route('/train')
def about():
    return render_template('train.html')

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

def train(folder_path, batch_size, num_epochs):
    shutil.unpack_archive(folder_path, "./dataset/")
    dls = ImageDataLoaders.from_folder(path=Path("dataset/train/"),shuffle=True, item_tfms=RandomResizedCrop(128, min_scale=0.35), valid_pct=0.2)
    learn = cnn_learner(dls, resnet50, metrics=[accuracy, error_rate])
    learn.fine_tune(int(num_epochs))
    with open('file.pkl', 'wb') as file:
        pickle.dump(learn, file)
    learn.save('final_model')
    return render_template("inference.html")

    

@app.route('/begin', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        epochs = request.form['epochs']
        batch = request.form['batch']
        print("Epochs:", epochs)
        print("Batch:", batch)
        f = request.files['file']
        f.save(f.filename)
        train(f.filename, batch, epochs)
        return render_template("inference.html")


# ******************* VISUALIZATION *******************

@app.route('/netron', methods=["POST"])
def index():
    if request.method == "POST":
        model_file = request.files["weights"]
        if model_file is None or model_file == '':
            model_file = "./final_model.pth"
        logging.info(f"model file path: {model_file}")
        # path = Path(model_file.filename)
        # model_file_suffix = path.suffix
        # if not (path.exists() and path.is_file()):
        #     return render_template('error.html')

        # if model_file_suffix not in file_type_list:
        #     return render_template('error.html')
        return render_template('netron.html', model_file="dataset/train/models/"+model_file.filename)

@app.route('/downloadfile', methods=['GET'])
def downloadfile():
    store_path = request.args.get('modelFile')
    logging.info(f"model download path: {store_path}")
    def send_file():
        with open(store_path, 'rb') as targetfile:
            while 1:
                data = targetfile.read(20 * 1024 * 1024)
                if not data:
                    break
                yield data

    path = Path(store_path)
    model_name = path.name
    response = Response(send_file(), content_type='application/octet-stream')
    response.headers["Content-disposition"] = 'attachment; filename=%s' % model_name
    response.headers["filename"] = model_name
    return response

# ******************* VISUALIZATION *******************



# ******************* INFERENCE *******************










# ******************* INFERENCE *******************

if __name__ == '__main__':

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    port = int(os.environ.get('PORT', 5100))

    app.run(host='0.0.0.0', port=port)