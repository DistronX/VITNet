from flask import Flask,render_template,url_for,request,session,logging,redirect,flash
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session,sessionmaker
import os

engine=create_engine("mysql+pymysql://root:@localhost/users")
db=scoped_session(sessionmaker(bind=engine))


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.sqlite3'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)