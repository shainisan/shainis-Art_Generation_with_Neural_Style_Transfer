import os
import tensorflow as tf
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import numpy as np
import PIL.Image
import gradio as gr
import tensorflow_hub as hub
import matplotlib.pyplot as plt


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


style_urls = {
    'Kanagawa great wave': 'The_Great_Wave_off_Kanagawa.jpg',
    'Kandinsky composition 7': 'Kandinsky_Composition_7.jpg',
    'Hubble pillars of creation': 'Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg',
    'Van gogh starry night': 'Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
    'Turner nantes': 'JMW_Turner_-_Nantes_from_the_Ile_Feydeau.jpg',
    'Munch scream': 'Edvard_Munch.jpg',
    'Picasso demoiselles avignon': 'Les_Demoiselles.jpg',
    'Picasso violin': 'picaso_violin.jpg',
    'Picasso bottle of rum': 'picaso_rum.jpg',
    'Fire': 'Large_bonfire.jpg',
    'Derkovits woman head': 'Derkovits_Gyula_Woman_head_1922.jpg',
    'Amadeo style life': 'Amadeo_Souza_Cardoso.jpg',
    'Derkovtis talig': 'Derkovits_Gyula_Talig.jpg',
    'Kadishman': 'kadishman.jpeg'
}


style_images = [k for k, v in style_urls.items()]


content_image_input = gr.inputs.Image(label="Content Image")
radio_style = gr.Radio(style_images, label="Choose Style")


def perform_neural_transfer(content_image_input, style_image_input):

    content_image = content_image_input.astype(np.float32)[np.newaxis, ...] / 255.
    content_image = tf.image.resize(content_image, (400, 600))
    
    style_image_input = style_urls[style_image_input]
    style_image_input = plt.imread(style_image_input)
    style_image = style_image_input.astype(np.float32)[np.newaxis, ...] / 255.
    
    style_image = tf.image.resize(style_image, (256, 256))

    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    return tensor_to_image(stylized_image)


app_interface = gr.Interface(fn=perform_neural_transfer,
                             inputs=[content_image_input, radio_style],
                             outputs="image",
                             title="Art Generation with Neural Style Transfer",
                            )
app_interface.launch()
