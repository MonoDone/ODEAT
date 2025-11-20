<div align="center">
  <h3>ODEAT: Ensemble Adversarial Training for Object Detection</h3>
</div>


<h2 id="quick-start">Quick Start</h2>

This is the official implementation for [ODEAT: Ensemble Adversarial Training for Object Detection]().


<h3>Preparation</h3>

  ```sh
  conda create -n oddefense python=3.10
  conda activate oddefense

  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

  pip install -U openmim
  mim install mmcv-full==1.7.0
  pip install mmdet==2.28.0
  pip install -r requirements.txt
  ```

  Download pretrained ResNet-50 backbone: <a href='https://huggingface.co/suixin1424/oddefense/blob/main/resnet50_linf_eps4_pure.pth'>resnet-50 pretrained </a>
  
  Download pretrained ConvNeXt-T backbone: <a href='https://huggingface.co/suixin1424/oddefense/blob/main/convnext_tiny_mmcls-linf-eps-4-advan.pth'>convnext-t pretrained </a>
  
  Download pretrained SwinTransformer-Base backbone: <a href='https://huggingface.co/suixin1424/oddefense/blob/main/convnext_tiny_mmcls-linf-eps-4-advan.pth'>convnext-t pretrained </a>

  

<h3>Train and Evaluate</h3>

1. **Modify Config Files**  
   Update the following variables in the config files (e.g., `frcnn/faster_rcnn_r50_fpn_1x_coco_freeat_all.py`):
   - `checkpoint_at`
   - `data_root`
   - `work_dir`

2. **Training**  
   Run the following command to start training:
    ```bash
    bash tools/dist_train_ensemble.sh [config_file] [num_gpus]
    ```

    If you want to change ensemble models, please modify the train file [`mmdet/tools/train_adv_ensemble.py`](mmdet/tools/train_adv_ensemble.py).

3. **Evaluation**  
  Run the following command to evaluate your model:
    ```bash
    bash tools/dist_test.sh [config_file] [ckpt_path] [num_gpus] --eval bbox
    ```

4. **Adversarial Examples**
  Run the following command to generate adversarial examples:
    ```bash
    bash tools/dist_test2.sh [config_file] [ckpt_path] [num_gpus] --eval bbox
    ```

    Config files are in [`coco/black_configs`](coco/black_configs).

5. **Check Model's Robustness**
  Run the following command to check the model's robustness on adversarial examples:
    ```bash
    bash tools/dist_test3.sh [config_file] [ckpt_path] [num_gpus] --eval bbox
    ```
  
    Before running this, you need to modify the `single_gpu_test` and `multi_gpu_test` functions in `mmdet/apis/test3.py`.
  
    Set the variable `adv_load_dir` to **your adversarial example folder**, for example: adv_load_dir = "/path/to/your/adversarial_examples"


<h2 id="models">Models</h2>

| **Model**       | **Config File**                                                                                     | **Checkpoint**                          |
|------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------|
| Faster-RCNN-ENSEMBLE  | [`faster_rcnn_r50_fpn_1x_coco_freeat_all.py`](frcnn/faster_rcnn_r50_fpn_1x_coco_freeat_all.py)            | <a href='https://pan.baidu.com/s/1Ip3rHBI-wRI_LZzCFX-F6g?pwd=mono'> click to download </a> |
| FCOS-ENSEMBLE            | [`fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_all.py`](fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_all.py)                                       | <a href='https://pan.baidu.com/s/1UblaSKf2i_EFrBgs_f9luA?pwd=mono'> click to download </a>            |
| DN-DETR-ENSEMBLE         | [`dn_detr_r50_8x2_12e_coco_freeat_all.py`](dn_detr/dn_detr_r50_8x2_12e_coco_freeat_all.py)                                   | <a href='https://pan.baidu.com/s/1Qkea_uHt9OLfTZzYZmsiYw?pwd=mono'> click to download </a>         |
| Faster-RCNN ConvNeXt-ENSEMBLE   | [`faster_rcnn_convnext_fpn_1x_coco_freeat_all.py`](frcnn/faster_rcnn_convnext_fpn_1x_coco_freeat_all.py)                                   | <a href='https://pan.baidu.com/s/1MtvCIfEAfo6CvHYAwoytsg?pwd=mono'> click to download </a>         |
| FCOS ConvNeXt-ENSEMBLE     | [`fcos_convnext_caffe_fpn_gn-head_1x_coco_freeat_all.py`](fcos/fcos_convnext_caffe_fpn_gn-head_1x_coco_freeat_all.py)                                       | <a href='https://pan.baidu.com/s/1iU1XvIAlA5vocnpcy9w5TA?pwd=mono'> click to download </a>            |
| DN-DETR ConvNeXt-ENSEMBLE   | [`dn_detr_convnext_8x2_12e_coco_freeat_all.py`](dn_detr/dn_detr_convnext_8x2_12e_coco_freeat_all.py)                                   | <a href='https://pan.baidu.com/s/1HlmAMKd-tINxTK8EATO8Sw?pwd=mono'> click to download </a>         |
| Faster-RCNN-TWO-ENSEMBLE   | [`faster_rcnn_r50_fpn_1x_coco_freeat_all.py`](frcnn/faster_rcnn_r50_fpn_1x_coco_freeat_all.py)            | <a href='https://pan.baidu.com/s/1iLQemSDTuqIWhBNLksgAGA?pwd=mono'> click to download </a> |
| Faster-RCNN-ENSEMBLE-PROCESSING   | [`faster_rcnn_r50_fpn_1x_coco_freeat_all.py`](frcnn/faster_rcnn_r50_fpn_1x_coco_freeat_all.py)            | <a href='https://pan.baidu.com/s/1c-eVyavTA88usJvrF_0tKQ?pwd=mono'> click to download </a> |

### Acknowledgement

This repository is built upon and modified from the open-source project
[oddefense](https://github.com/thu-ml/oddefense) by Tsinghua University Machine Learning Group.
We sincerely thank the authors for releasing their code.

If you use this code in your research, please also consider citing the original oddefense work in addition to ours.

