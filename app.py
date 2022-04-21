from distutils.log import debug
from flask import Flask,render_template,url_for,request,session,logging,redirect,flash
from flask import send_file
# from sqlalchemy import create_engine
# from sqlalchemy.orm import scoped_session,sessionmaker
import os
from fastai.vision.all import *
import shutil

UPLOAD_FOLDER = '.'

# engine=create_engine("mysql+pymysql://root:@localhost/users")
# db=scoped_session(sessionmaker(bind=engine))


app = Flask(__name__)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.sqlite3'


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/infer')
def infer():
    return render_template('inference.html')

@app.route('/visualize')
def visualize():
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
    dls = ImageDataLoaders.from_folder(path=Path("./dataset/"),bs=batch_size,shuffle=True, item_tfms=RandomResizedCrop(128, min_scale=0.35))
    learn = cnn_learner(dls, resnet50, metrics=[accuracy, error_rate])
    learn.fine_tune(num_epochs)
    learn.export('yourmodel.pkl')
    return send_file('./yourmodel.pkl', as_attachment=True)

    

@app.route('/begin', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        epochs = request.form['epochs']
        batch = request.form['batch']
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(url_for('infer'))
        file = request.files['file']
        if file.filename == '':
            #flash('No selected file')
            return redirect(url_for('infer'))
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            train(file_path, batch, epochs)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    port = int(os.environ.get('PORT', 5100))
    app.run(host='0.0.0.0', port=port)