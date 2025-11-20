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

4. **Check Model's Robustness**
  Run the following command to check the model's robustness on adversarial examples:
    ```bash
    bash tools/dist_test3.sh [config_file] [ckpt_path] [num_gpus] --eval bbox
    ```
  
  Before running this, you need to modify the `single_gpu_test` and `multi_gpu_test` functions in `mmdet/apis/test3.py`.
  Set the variable `adv_load_dir` to **your adversarial example folder**, for example:
    ```python
    adv_load_dir = "/path/to/your/adversarial_examples"
    ```


<h2 id="models">Models</h2>

| **Model**       | **Config File**                                                                                     | **Checkpoint**                          |
|------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------|
| Faster-RCNN  | [`faster_rcnn_r50_fpn_1x_coco_freeat_all.py`](frcnn/faster_rcnn_r50_fpn_1x_coco_freeat_all.py)            | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/frcnn_at.pth'> click to download </a> |
| FCOS            | [`fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_all.py`](fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_all.py)                                       | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/fcos_at.pth'> click to download </a>            |
| DN-DETR         | [`dn_detr_r50_8x2_12e_coco_freeat_all.py`](dn_detr/dn_detr_r50_8x2_12e_coco_freeat_all.py)                                   | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/dndetr_at.pth'> click to download </a>         |
| Faster-RCNN ConvNeXt   | [`faster_rcnn_convnext_fpn_1x_coco_freeat_all.py`](frcnn/faster_rcnn_convnext_fpn_1x_coco_freeat_all.py)                                   | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/frcnn_convnext.pth'> click to download </a>         |
| FCOS ConvNeXt     | [`fcos_convnext_caffe_fpn_gn-head_1x_coco_freeat_all.py`](fcos/fcos_convnext_caffe_fpn_gn-head_1x_coco_freeat_all.py)                                       | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/fcos_convnext.pth'> click to download </a>            |
| DN-DETR ConvNeXt   | [`dn_detr_convnext_8x2_12e_coco_freeat_all.py`](dn_detr/dn_detr_convnext_8x2_12e_coco_freeat_all.py)                                   | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/dndetr_convnext.pth'> click to download </a>         |

### Acknowledgement

This repository is based on the official implementation of the following works.  
If you find this code useful in your research, please also consider citing:

```bibtex
@article{li2025importance,
  title   = {On the importance of backbone to the adversarial robustness of object detectors},
  author  = {Li, Xiao and Chen, Hang and Hu, Xiaolin},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2025},
  publisher = {IEEE}
}

@inproceedings{li2025pbcat,
  title     = {PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection},
  author    = {Li, Xiao and Zhu, Yiming and Huang, Yifan and Zhang, Wei and He, Yingzhe and Shi, Jie and Hu, Xiaolin},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
···
