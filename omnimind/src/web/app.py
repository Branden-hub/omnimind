from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from functools import wraps
from omnimind import OmniMind
import ssl
import config
import secrets
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config.from_object('config')
omnimind = OmniMind()

# Set up logging
if not app.debug:
    file_handler = RotatingFileHandler('omnimind.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('OmniMind startup')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if (request.form['username'] == app.config['USERNAME'] and 
            request.form['password'] == app.config['PASSWORD']):
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('t1.html')

@app.route('/quantum')
@login_required
def quantum():
    return render_template('t2.html')

@app.route('/neural')
@login_required
def neural():
    return render_template('t3.html')

@app.route('/knowledge')
@login_required
def knowledge():
    return render_template('t4.html')

@app.route('/agent-builder')
@login_required
def agent_builder():
    return render_template('agent_builder.html')

@app.route('/component-builder')
@login_required
def component_builder():
    return render_template('component_builder.html')

# ... [previous API routes remain the same, just add @login_required decorator]

if __name__ == '__main__':
    if app.config['SSL_ENABLED']:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(app.config['SSL_CERT'], app.config['SSL_KEY'])
        app.run(
            host=app.config['HOST'],
            port=app.config['PORT'],
            ssl_context=context,
            debug=False  # Set to False in production
        )
    else:
        app.run(
            host=app.config['HOST'],
            port=app.config['PORT'],
            debug=False  # Set to False in production
        )
