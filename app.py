import os

import gradio as gr
import os
import random
from PIL import Image
import time
import torch
from torch import autocast
from kandinsky2 import get_kandinsky2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_kandinsky2(
    'cuda', 
    task_type='text2img', 
    cache_dir='/media/agp/d58e0f56-1cd5-45af-938c-27e43b4fc343/kandinsky/tmp', 
    model_version='2.1', 
    use_flash_attention=False
)


"""
num_steps=50, 
    batch_size=4, 
    guidance_scale=7,
    h=768, 
    w=768,
    sampler='ddim_sampler', 
    prior_cf_scale=1,
    prior_steps='25',
"""
#@title Images Generation

    
def infer(prompt, negative,h,w,batch_size, num_steps, guidance_scale,prior_cf_scale):
    images = model.generate_text2img(prompt, 
                           negative_prior_prompt=negative,
                           negative_decoder_prompt=negative, 
                           num_steps=int(num_steps),
                           batch_size=int(batch_size),
                           guidance_scale=int(guidance_scale),
                           h=int(h), w=int(w),
                           sampler='p_sampler',
                           prior_cf_scale=int(prior_cf_scale),
                           prior_steps='30',)
    return images
def image_1 (prompt, negative,h,w,batch_size, num_steps, guidance_scale,prior_cf_scale) 
image_1 = model.generate_text2img(
    prompt,
    num_steps=100,
    batch_size=1,
    guidance_scale=4,
    h=768,
    w=768,
    sampler='p_sampler', 
    prior_cf_scale=4,
    prior_steps="25"
)[0]
def image_2 (prompt, negative,h,w,batch_size, num_steps, guidance_scale,prior_cf_scale) 
image_1 = model.generate_text2img(
    prompt,
    num_steps=100,
    batch_size=1,
    guidance_scale=4,
    h=768,
    w=768,
    sampler='p_sampler', 
    prior_cf_scale=4,
    prior_steps="25"
)[0]
  
image_mixed = model.mix_images(
    [image_1, image_2], [0.5, 0.5], 
    num_steps=100, 
    batch_size=1, 
    guidance_scale=4, 
    h=768, 
    w=768, 
    sampler='p_sampler', 
    prior_cf_scale=4, 
    prior_steps="5",
)[0]

css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 930px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        #container-advanced-btns{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }

        }
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #generated_id{
            min-height: 1860x
        }
"""
block = gr.Blocks(css=css)


#SPACE_ID = os.getenv('SPACE_ID')

with block as webui:
    gr.Markdown(f"""
        """
    )
    with gr.Tab("text to image"):
        with gr.Group():
            with gr.Box():
                with gr.Row():

                    text = gr.Textbox(
                        label="Enter your prompt", show_label=True, max_lines=2
                    )
                    negative = gr.Textbox(
                        label="Enter your negative prompt", show_label=True, max_lines=2
                    )

                with gr.Row():
                    with gr.Accordion("Advanced image settings", open=False):
                        h =  gr.Slider(minimum=512, maximum=1280, step=64 ,label="Height.  Minimum 512px, maximum 1280px")  
                        w = gr.Slider(minimum=512, maximum=1280, step=64, label="Width. Minimum 512px , Maximum 1280px")
                        num_steps = gr.Slider(minimum=40, maximum=150, step=5 ,label="Number of Steps:  Minimum 30, maximum 150")
                        batch_size = gr.Slider(minimum=1, maximum=8, step=1, label="Number of images to generate:  Minimum 1, maximum 8")  
                        guidance_scale = gr.Slider( minimum=1, maximum=20, step=1, label="Guidance scale. A high guidance scale means that the model should generate images that closely match the specified style or theme, while a low guidance scale allows the model to generate more diverse and original images")
                        prior_cf_scale = gr.Slider(minimum=1, maximum=20, step=1, label="Prior config scale.Overall, the prior config scale hyperparameter allows users to control the level of adherence to specified conditions in the generated images. A high prior config scale results in images that closely match the specified conditions, while a low prior config scale generates more diverse and creative images.")
                       #prior_steps = gr.Slider(minimum=1, maximum=50, step=1, label="Prior steps.  Increasing the prior steps can result in more detailed and accurate output, but it can also make the model slower and more computationally expensive.")


                with gr.Row():
                    btn = gr.Button("Generate")

            gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="generated_id").style(columns=[2], rows=[2], object_fit="contain", height="auto")

           # ex = gr.Examples(examples=examples, fn=infer, inputs=[text, negative], outputs=gallery, cache_examples=True)
            #ex.dataset.headers = [""]

            text.submit(infer, inputs=[text, negative, h, w, batch_size, num_steps, guidance_scale, prior_cf_scale], outputs=gallery)
            btn.click(infer, inputs=[text, negative, h, w, batch_size, num_steps, guidance_scale, prior_cf_scale], outputs=gallery)
    with gr.Tab("Flip Text"):
            gr.Markdown(f"""
                text test
        """
    )
  

webui.queue(max_size=15).launch()