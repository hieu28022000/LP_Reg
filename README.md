
<h1 align="center">License plate OCR</h1>



<br>

## About ##

This project do about vietnam license plates recognition, we use a self-build dataset for training detect & recognition model

At License plate detect we use YOLOv4

At Text detection on License plate we use CRAFT

At Text recognition on License plate we use Star-Net

## Starting ##

```bash
# Clone this project
$ git clone https://github.com/hieu28022000/LP_Reg.git

# Access
$ cd LP_reg

# Install dependencies
$ pip install -r requirements.txt

# download pretrain model
./download_model.sh

# unzip pretrain models
unzip models.zip
```
## Run the project ##

```bash
# run pipeline on images folder
python3 pipeline.py --run_on_folder True --folder_path <path to folder image>
# or run pipeline on image
python3 pipeline.py --run_on_folder False --image_path <path to image>
```

# Result # 
You can see full project and result in [here](https://drive.google.com/file/d/1ge7vQjHqn9bDM_3kKxZk7B_t62dN5DaI/view?usp=sharing)

<a href="#top">Back to top</a>
