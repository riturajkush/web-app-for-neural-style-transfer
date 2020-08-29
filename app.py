import tensorflow as tf
#tf.enable_eager_execution()
import streamlit as st
import io
import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import Model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import time



#PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
#st.beta_set_page_config(**PAGE_CONFIG)

def main(style, content):





    #@st.cache(suppress_st_warning=True)
    def load_and_process_image(img):
        #img = load_img(image_path)
        #print(img)
        #img = Image.open(img)
        #st.image(img, width=None)
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis = 0)
        return img


    content = load_and_process_image(content)
    style = load_and_process_image(style)


    #@st.cache(suppress_st_warning=True)
    def deprocess(x):
    # perform the inverse of the preprocessiing step
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    #@st.cache(suppress_st_warning=True)
    def display_image(image):
        if len(image.shape) == 4:
            img = np.squeeze(image, axis = 0)

        img = deprocess(img)

        #plt.grid(False)
        #plt.xticks([])
        #plt.yticks([])
        plt.figure()
        plt.imshow(img, interpolation='nearest')
        #img = Image.open(io.StringIO(img))
        #st.image(img)
        st.pyplot()

    model = VGG19(include_top = False, weights = 'imagenet')

    model.trainable = False

    style_layers = ['block1_conv1', 'block3_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    content_model = Model(inputs = model.input, outputs = model.get_layer(content_layer).output)

    style_models = [Model(inputs = model.input, outputs = model.get_layer(layer).output) for layer in style_layers]

    #@st.cache(suppress_st_warning=True)
    def content_cost(content, generated):
        a_C = content_model(content)
        a_G = content_model(generated)
        cost = tf.reduce_mean(tf.square(a_C - a_G))
        return cost

    #@st.cache(suppress_st_warning=True)
    def gram_matrix(A):
        channels = int(A.shape[-1])
        a = tf.reshape(A, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a = True)
        return gram / tf.cast(n, tf.float32)

    lam = 1. / len(style_models)

    #@st.cache(suppress_st_warning=True)
    def style_cost(style, generated):
        J_style = 0
        for style_model in style_models:
            a_S = style_model(style)
            a_G = style_model(generated)
            GS = gram_matrix(a_S)
            GG = gram_matrix(a_G)
            current_cost = tf.reduce_mean(tf.square(GS - GG))
            J_style += current_cost * lam

        return J_style

    generated_images = []




    #@st.cache(suppress_st_warning=True)
    def training_loop(content, style, iterations = 10, a = 10., b = 20.):
        # initialise
        #content = load_and_process_image(content_path)
        #style = load_and_process_image(style_path)
        #st.subheader("@@")
        generated = tf.Variable(content, dtype = tf.float32)
        opt = optimizers.Adam(lr = 7.)

        best_cost = 1e12+0.1
        best_image = None

        start_time = time.time()
        my_bar = st.progress(0)
        for i in range(iterations):

            with tf.GradientTape() as tape:
                J_content = content_cost(content, generated)
                J_style = style_cost(style, generated)
                J_total = a * J_content + b * J_style

            grads = tape.gradient(J_total, generated)
            opt.apply_gradients([(grads, generated)])

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated.numpy()

            if i % int(iterations/10) == 0:
                time_taken = time.time() - start_time
                my_bar.progress(i+91)
                #st.subheader('Cost at {}: {}. Time elapsed: {}'.format(i, J_total, time_taken))
                generated_images.append(generated.numpy())

        return best_image









    if((content is not None) and (style is not None)):
        #st.subheader("$$$")
        st.write("")
        st.write("Processing...")

        final = training_loop(content, style)
        #final = Image.open(final)
        #st.image(final, width=None)
        display_image(final)

    #st.header("We're done")
    st.balloons()





if __name__ == '__main__':
    tf.keras.backend.clear_session()

    st.title("Neural Style Transfer")
    st.subheader("Try your image in different styles. Have fun 😎 ")

    con = 0
    sty = 0
    #k = 0
    img_file_buffer1 = st.file_uploader("Upload a content image", type=["png", "jpg", "jpeg"], key="content")
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    if(img_file_buffer1 is not None):
        content = Image.open(img_file_buffer1)
        st.image(content, width=None)
        content = content.resize((150,150))
        con = 1

    img_file_buffer2 = st.file_uploader("Upload a style image", type=["png", "jpg", "jpeg"], key="style")
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    if(img_file_buffer2 is not None):
        style = Image.open(img_file_buffer2)
        st.image(style, width=None)
        style = style.resize((150,150))
        sty = 1
        #k += 1

    if(con==1 and sty==1):
        main(style, content)
