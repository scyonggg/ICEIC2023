# 360former_project



## 1.Installation

### 1) Setting environment
Install anaconda according to the server environment.
Refer to the following command to create the environment from yaml file.

~~~bash
conda env create --file environment.yml
~~~

### 2) Prepare dataset

For training, we will use **Structured3d** dataset.

For evaluation, we will use **Stanford, Pano3D, Structured3d** dataset.

**For the time being, prepare Structured3d dataset first for each own server following the instructions below.**

* [Stanford](https://github.com/alexsax/2D-3D-Semantics): Create train/val/test split following their instructuions. Data folder should be constructed as below. Refer to their repositoiries for more details.

```bash
├── Stanford
   ├── area_n
        ├── rgb
            ├── image.png
        ├── depth
            ├── depth.png
       
``` 
* [Structure3D](https://github.com/bertjiazheng/Structured3D) : Create train/val/test split following their instructuions. Data folder should be constructed as below. Refer to their repositoiries for more details.

```bash
├── Structure3D
   ├── scene_n
        ├── 2D_rendering
            ├── room_number
                ├── panorama
                    ├── full
                        ├── rgb_rawlight.png
                        ├── depth.png
``` 
* [Pano3D](https://github.com/alexsax/2D-3D-Semantics): Download [Matterport3D Train & Test (/w Filmic) High Resolution (1024 x 512)](https://zenodo.org/record/5707345#.YZY3-2BByUk) following the instructions in [Pano3D project page](https://github.com/alexsax/2D-3D-Semantics)

## 2.Quick start (Multi-GPU Training)
Open the **script** file, create checkpoint folder and modify the **--model_name** configuration according to the created checkpoint folder path. Also, select the GPU ID(s) to use (e.g. '0,1') and modify the **--gpu** configuration (e.g. `--gpu 0,1`)

For more details about configuration, refer to **main.py**.

~~~bash
bash script
~~~

## 3.Issues
Please check the issues frequently, and reply to those issues.
Please upload any minor issues regarding codes. 

If an error message `RuntimeError: Address already in use` appears, please open the **script** file and modify configuration `--dist_url` into another port (e.g. --dist_url "tcp://127.0.0.1:7890")