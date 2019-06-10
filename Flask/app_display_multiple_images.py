import os


from flask import Flask, request, render_template, send_from_directory,session

from tensorflow.python.client import device_lib 

import Modelo256_v2 as md
import tensorflow as tf
import cv2 as cv2
import scipy.misc as sm
import mascaraClass as MC
import numpy as np

app = Flask(__name__)



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    
    
x_input_shape=(1,256,256,3)
x_input=tf.placeholder(tf.float32,shape=x_input_shape,name="ph_X_input")
            
    #modelo 
modeloBuilder=md.Model(1)
with tf.variable_scope('Model') as scope:
    model=modeloBuilder.buildModel(x_input)
        
        
    #varibles
variables=tf.trainable_variables()
G_vars=[ var.name for var in variables]
variables=dict(zip(G_vars,variables))
model_nameG="modelo_ED_D_v17G_2"
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver(variables)
saver.restore(sess, './model/cifar_'+model_nameG+'.ckpt')
mascaraC=MC.MaskClass()

@app.route("/")
def index():
    
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        
        
        
        imagen=sm.imread(destination)[:,:,:3]
        imagen=np.array(cv2.resize(imagen,(256,256)))
        mascaraD,mascaraD2=mascaraC.detectarMask(imagen)
        imagen=np.multiply(imagen,mascaraD,dtype=np.uint8)
        
        imagenes=np.expand_dims(imagen, axis=0).copy()
        train_dict = {x_input: imagenes}
        predcicciones=sess.run(model,feed_dict=train_dict)
        predcicciones=np.array(predcicciones,dtype=np.uint8)
        mascaraD,mascaraD2=mascaraC.detectarMask(imagen)
        resultado=(imagen*mascaraD)+(mascaraD2*predcicciones[0])
        destination = "/".join([target, filename+"_R_.png"])
        sm.imsave(destination,resultado)
        session["resultado"]=resultado
        

    # return send_from_directory("images", filename, as_attachment=True)
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)



if __name__ == "__main__":
    app.run()