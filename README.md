# kandinsky web user interface
small Gradio web user interface used for generating images 
Modify the line number 17  and edit the path where the model is saved

    cache_dir='/media/agp/d58e0f56-1cd5-45af-938c-27e43b4fc343/kandinsky/tmp', 
    
-------------------
python 3.10.6 has to be installed.
you can download an install from : https://www.python.org/downloads/release/python-3106/



Install steps :

1. mkdir kandinsky && cd kandinsky
2. python -m venv venv
3. source venv/bin/activate
4. git clone https://github.com/ai-forever/Kandinsky-2.git
5. cd Kandinsky-2
6.  pip install "git+https://github.com/ai-forever/Kandinsky-2.git"
7. pip install git+https://github.com/openai/CLIP.git
8. pip install gradio
9. wget https://raw.githubusercontent.com/adrianpuiu/kandinsky/main/app.py
10 python app.py 

open and point your browser at : http://127.0.0.1:7860

Enjoy generating images

-----------------------------------------------------------
If you also want to access inpaint and image mix features you'll need to install jupyterlab module 
-------------
1. pip install jupyterlab
2. jupiter-lab 
3. open the browser and go to notebooks folder and select the one you want
-------------
