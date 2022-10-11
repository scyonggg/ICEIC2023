# Evaluation

#### 1) Download pre-trained models
* Pre-trained Bifuse model : will be uploaded.
* Pre-trained Hohonet model : will be uploaded
* Pre-trained Joint_360depth model : Refer to [Joint_360depth](https://github.com/yuniw18/Joint_360depth)

#### 2) Prepare the test dataset (Pano3D testset will be added)
* [3D60 data](https://github.com/VCL3D/3D60) (Stanford3D, Matterport3D, SunCG): Create train/val/test split following their instructuions.   We only use Center view for 3D60 dataset.   Refer to the sample test split of Stanford3D data in 3D60_split folder.
* [Stanford data](https://github.com/alexsax/2D-3D-Semantics): Create train/val/test split following their instructuions. Data folder should be constructed as below. Refer to their repositoiries for more details.

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

#### 3) Go to evaluate folder & run the following command
For detailed command options, please refer to `evaluate_main.py` and `eval_script`.

~~~bash
cd evaluate
python3 evaluate_main.py --method [method to be tesetd]  --eval_data [test_data_name] --data_path [test_data_path] --checkpoint_path [checkpoint_path]
~~~
