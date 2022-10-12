# Evaluation



#### 1) Prepare the test dataset (Currently, Structured3D dataset is only tested.)

* [Structure3D](https://github.com/bertjiazheng/Structured3D) : **test split scene # 03250 ~ 03499**.

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

* [Pano3D](https://github.com/alexsax/2D-3D-Semantics): Download [Matterport3D Train & Test (/w Filmic) High Resolution (1024 x 512)](https://zenodo.org/record/5707345#.YZY3-2BByUk) following the instructions in [Pano3D project page](https://github.com/alexsax/2D-3D-Semantics). Test splits are included in ``` pano_loader```  folder

* [3D60 data](https://github.com/VCL3D/3D60) (Stanford3D, Matterport3D, SunCG): Create train/val/test split following their instructuions.   We only use Center view for 3D60 dataset.   
* [Stanford data](https://github.com/alexsax/2D-3D-Semantics): Create train/val/test split following their instructuions. Data folder should be constructed as below. Refer to their repositoiries for more details.

```bash
├── Stanford
   ├── area_n
        ├── rgb
            ├── image.png
        ├── depth
            ├── depth.png
       
``` 

#### 2) Go to evaluate folder & run the `eval_script`
For detailed command options, please refer to `evaluate_main.py` and **`eval_script`**.

~~~bash
bash eval_script
~~~

For Joint_360depth model (download **Super_S3D_Fres.pth** pretrained model), check if the following results are extracted.

~~~bash
Avg.Abs.Rel.Error: 0.0368
Avg.Sq.Rel.Error:0.0022
Avg.Lin.RMS.Error:0.0268
Avg.Log.RMS.Error:0.0880
Inlier D1:0.9825
Inlier D2: 0.9945
Inlier D3: 0.9973
RMSE: 0.0231

~~~
# Experimental results
* The person in charge (PIC) of each experiments ID should upload the pretrained model in the shared folder of INHA server (```\mnt/prj/users/depth/360former_experiments```) to make it reproducible. 
* Please link the commit of the model (used for training) and evaluation as shown table below. For evaluation link, update the eval_script and codes to be executable via simple command line (e.g., ``` bash eval_script```)
* Put :white_check_mark: if all things are done , else put 🔲:

##  Table 1 
Structure3D training / **Image-align based evaluation**

|ID| Model               | BackBone | checkpoints | Commit <br>(model/evaluation) |Abs. rel. | Sq.rel | Lin.rms | delta < 1.25  | PIC |
|----|---------------------|--------------------|----------------|--------------------------|-----------------|------|------|----------------|-----------|
|1| Joint_depth     | ViT |   Super_S3D_Fres.pth |[a01617b](https://github.com/yuniw18/Joint_360depth/commit/a01617bc9f0579ae70c108074ce6030d3785c1ab) / [c0db391](https://github.com/yuniw18/360former_project/tree/c0db391e10722ebc850ce247a43a683d4c6a5e18)| 0.0368    | 0.0022     | 0.0268 |0.9825|:white_check_mark: Yun|
|2| Joint_depth     | ViT |   Joint_S3D_Fres.pth |[a01617b](https://github.com/yuniw18/Joint_360depth/commit/a01617bc9f0579ae70c108074ce6030d3785c1ab)/[c0db391](https://github.com/yuniw18/360former_project/tree/c0db391e10722ebc850ce247a43a683d4c6a5e18)| 0.0415    | 0.0026     | 0.0291 |0.9809|:white_check_mark: Yun|
|3| Bifuse     | - |  gen_latest.pkl  |[6fb1cbe](https://github.com/Yeh-yu-hsuan/BiFuse/commit/6fb1cbe8a3c3891a9067f595ba2af9d14f8ae1c6) / TBU| 0.0571 |  0.0048    | 0.0386 |0.9666|:black_square_button: Yun|
|4| HohoNet     | - |  gen_latest.pkl |[5ad7f48](github.com/sunset1995/HoHoNet/commit/5ad7f486a26b13834abee61527ad2aaa98ff6fbe)/TBU |0.0789|  0.0081 |  0.0473 | 0.9411 |:black_square_button: Yun|
|5| Baseline | CSwin |  epoch_15   | [c0db391](https://github.com/yuniw18/360former_project/tree/c0db391e10722ebc850ce247a43a683d4c6a5e18)/ [c0db391](https://github.com/yuniw18/360former_project/tree/c0db391e10722ebc850ce247a43a683d4c6a5e18) |0.1578 | 0.0150  | 0.0723   | 0.8073|:white_check_mark: Yun|

##  Table 2 
Structure3D training / **Column-align based evaluation**

|ID| Model               | Backbone | checkpoints name | Commit <br>(model/evaluation) |Abs. rel. | Sq.rel | Lin.rms | delta < 1.25  | PIC |
|----|---------------------|--------------------|----------------|--------------------------|-----------------|------|------|----------------|-----------|
|1| Joint_depth     | ViT |   Super_S3D_Fres.pth |[a01617b](https://github.com/yuniw18/Joint_360depth/commit/a01617bc9f0579ae70c108074ce6030d3785c1ab) / [c0db391](https://github.com/yuniw18/360former_project/tree/c0db391e10722ebc850ce247a43a683d4c6a5e18)| 0.0265    | 0.0015     | 0.0201 |0.9886|:white_check_mark: Yun|
|2| Joint_depth     | ViT |   Joint_S3D_Fres.pth |[a01617b](https://github.com/yuniw18/Joint_360depth/commit/a01617bc9f0579ae70c108074ce6030d3785c1ab) / [c0db391](https://github.com/yuniw18/360former_project/tree/c0db391e10722ebc850ce247a43a683d4c6a5e18)| 0.0310    | 0.0017     | 0.0220 |0.9868|:white_check_mark: Yun|
|3| Bifuse     | - |  gen_latest.pkl  |[6fb1cbe](https://github.com/Yeh-yu-hsuan/BiFuse/commit/6fb1cbe8a3c3891a9067f595ba2af9d14f8ae1c6)/TBU| TBU |  TBU    | TBU |TBU|:black_square_button:  Yun|
|4| HohoNet     | - |  gen_latest.pkl | [5ad7f48](github.com/sunset1995/HoHoNet/commit/5ad7f486a26b13834abee61527ad2aaa98ff6fbe)/TBU |TBU|  TBU | TBU | TBU |:black_square_button: Yun|
|5| Baseline | CSwin |  epoch_16   | [c0db391](https://github.com/yuniw18/360former_project/tree/c0db391e10722ebc850ce247a43a683d4c6a5e18)/[c0db391](https://github.com/yuniw18/360former_project/tree/c0db391e10722ebc850ce247a43a683d4c6a5e18) | 0.0341| 0.0025  | 0.0256  | 0.9806|:white_check_mark: Yun|


