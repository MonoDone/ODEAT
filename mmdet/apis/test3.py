# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time, copy

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from mmcv.parallel import scatter
import cv2, os
import numpy as np

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# os.environ["HF_HUB_READ_TIMEOUT"] = "120"

# import timm
# import torch.nn as nn
# from mmdet.models import BACKBONES

# @BACKBONES.register_module()
# class TIMMBackbone(nn.Module):
#     """
#     轻量封装：用 timm 的 features_only 输出多尺度特征。
#     默认选一个与 FPN 匹配 [256,512,1024,2048] 的模型（如 seresnext50_32x4d）。
#     """
#     def __init__(self, model_name='seresnext50_32x4d', pretrained=True, out_indices=(1,2,3,4), **kwargs):
#         super().__init__()
#         self.out_indices = out_indices
#         self.timm = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=out_indices)
#         # 校验通道数（可打印或断言）
#         self.out_channels = [f['num_chs'] for f in self.timm.feature_info]  # 全部 stages
#         self.selected_channels = [self.timm.feature_info[i]['num_chs'] for i in out_indices]
#         # 如果你**必须**是 [256,512,1024,2048]，可以这里 assert
#         # assert self.selected_channels == [256,512,1024,2048]

#     def init_weights(self):
#         # 已由 timm 处理，这里可以留空
#         pass

#     def forward(self, x):
#         feats = self.timm(x)   # list of tensors for selected out_indices
#         return tuple(feats)     # MMDet 期望 tuple


def cal_adv(model, img, adv_sample, img_transform, test_adv_cfg):
    step_size = test_adv_cfg.get("step_size", 8)
    epsilon   = test_adv_cfg.get("epsilon", 8)
    num_steps = test_adv_cfg.get("num_steps", 20)

    adv_type = test_adv_cfg.get("adv_type", "cls")
    assert adv_type in ["cls", "reg", "cwa", "dag", "ours", "mi_fgsm"]

    img_adv = img.detach().clone().float()
    img_adv.requires_grad_(True)

    if adv_type == "cwa":
        for m in adv_sample["img_metas"]:
            m["cwa"] = True

    for m in adv_sample["img_metas"]:
        m["adv_flag"] = True

    if adv_type == "mi_fgsm":
        decay     = test_adv_cfg.get("decay", 1.0)
        momentum  = torch.zeros_like(img_adv)  # no grad
        step_size = test_adv_cfg.get("step_size", 2)
        num_steps = test_adv_cfg.get("num_steps", 10)
        epsilon   = test_adv_cfg.get("epsilon", 16)

    # （可选）随机起点：img_adv = torch.clamp(img + torch.empty_like(img).uniform_(-epsilon, epsilon), 0.0, 255.0)

    for _ in range(num_steps):
        img_adv = img_adv.detach()
        img_adv.requires_grad_(True)

        # ⚠️ 确认这里就是完整且可微的预处理
        tmp = img_transform[0](img_adv)
        adv_sample["img"] = tmp

        loss_dict = model(**adv_sample, return_loss=True)
        loss_calc_type = "cls" if adv_type == "mi_fgsm" else adv_type

        if loss_calc_type == "cls":
            v = loss_dict["loss_cls"]
            loss = (torch.stack(v).sum() if isinstance(v, list) else v.sum())
        elif loss_calc_type == "reg":
            v = loss_dict["loss_bbox"]
            loss = (torch.stack(v).sum() if isinstance(v, list) else v.sum())
        elif loss_calc_type == "cwa":
            v_cls = loss_dict["loss_cls"]
            v_box = loss_dict["loss_bbox"]
            cls_loss = (torch.stack(v_cls).sum() if isinstance(v_cls, list) else v_cls.sum())
            reg_loss = (torch.stack(v_box).sum() if isinstance(v_box, list) else v_box.sum())
            loss = cls_loss + reg_loss
        else:
            raise NotImplementedError

        x_grad = torch.autograd.grad(loss, [img_adv], retain_graph=False, create_graph=False)[0]

        if adv_type == "mi_fgsm":
            # per-sample L1 归一化
            denom = x_grad.view(x_grad.size(0), -1).norm(p=1, dim=1).view(-1,1,1,1).clamp_min(1e-12)
            x_grad = x_grad / denom
            x_grad = x_grad + decay * momentum
            momentum = x_grad.detach()

        eta = torch.sign(x_grad) * step_size

        # L∞ 投影（像素域 0~255；如用 0~1，请相应调整 epsilon/step_size 与 clamp）
        img_adv = img_adv + eta
        img_adv = torch.max(torch.min(img_adv, img + epsilon), img - epsilon)
        img_adv = torch.clamp(img_adv, 0.0, 255.0)

    for m in adv_sample["img_metas"]:
        m["adv_flag"] = False
        if "cwa" in m:
            m["cwa"] = False

    return img_adv


def _save_adv_image(tensor_img, img_transform, filename, save_dir, save_ext=".png"):
    """
    tensor_img: (1,3,H,W) in *normalized* space
    img_transform: (to_norm, to_denorm)
    filename: 原图在数据集中的名字（如 "JPEGImages/001923.jpg"）
    """
    # 先反归一化到 0~255 RGB
    denorm = img_transform[1](tensor_img.detach())

    # NCHW -> HWC, RGB -> BGR for cv2
    np_img = denorm.squeeze(0).permute(1, 2, 0)\
                    .clamp(0, 255).cpu().numpy().astype('uint8')
    np_img = np_img[:, :, ::-1]  # RGB->BGR

    # 组合输出路径：保留 filename 的子目录结构
    base, _ = os.path.splitext(filename)           # e.g. "JPEGImages/001923"
    out_path = os.path.join(save_dir, base + save_ext)  # e.g. "<save_dir>/JPEGImages/001923.png"

    # 关键：为多级子目录创建文件夹
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 保存
    cv2.imwrite(out_path, np_img)
    return out_path


def apply_saved_adv_to_sample(sample_or_data, adv_dir, device):
    # --- 解包 metas，同你现有代码 ---
    def unwrap_metas(metas_field):
        if isinstance(metas_field, list):
            first = metas_field[0]
            return first._data[0] if hasattr(first, '_data') else first
        else:
            return metas_field._data[0]

    def set_img(sample_field, img_tensor):
        if isinstance(sample_field, list):
            first = sample_field[0]
            if hasattr(first, '_data'):
                first._data[0] = img_tensor
            else:
                sample_field[0] = img_tensor
        else:
            if hasattr(sample_field, '_data'):
                sample_field._data[0] = img_tensor
            else:
                return img_tensor

    metas = unwrap_metas(sample_or_data['img_metas'])
    meta0 = metas[0]

    # === 读原图（BGR->RGB, float32, HWC） ===
    ori_name = meta0['ori_filename']
    base, _ = os.path.splitext(ori_name)
    path = None
    for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        cand = os.path.join(adv_dir, base + ext)
        if os.path.exists(cand):
            path = cand; break
    if path is None:
        raise FileNotFoundError(f"Adv image for {ori_name} not found under {adv_dir}")

    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype('float32')

    # === 1) 按 test pipeline 的 Resize 复现几何到 img_shape ===
    # meta['img_shape'] 是 (H, W, 3)
    tgt_h, tgt_w = meta0['img_shape'][:2]
    if (img_rgb.shape[0], img_rgb.shape[1]) != (tgt_h, tgt_w):
        img_rgb = mmcv.imresize(img_rgb, (tgt_w, tgt_h), interpolation='bilinear')

    # === 2) 按 RandomFlip（若有）复现 ===
    if meta0.get('flip', False):
        direction = meta0.get('flip_direction', 'horizontal')
        if direction == 'horizontal':
            img_rgb = img_rgb[:, ::-1, :]
        elif direction == 'vertical':
            img_rgb = img_rgb[::-1, :, :]

    # === 3) Normalize ===
    mean = torch.tensor(meta0['img_norm_cfg']['mean'], device=device).view(1,1,3).cpu().numpy()
    std  = torch.tensor(meta0['img_norm_cfg']['std'],  device=device).view(1,1,3).cpu().numpy()
    img_norm = (img_rgb - mean) / std

    # === 4) Pad 到 pad_shape（注意：Normalize 之后 pad 值为 0）===
    pad_h, pad_w = meta0['pad_shape'][:2]
    if (tgt_h, tgt_w) != (pad_h, pad_w):
        canvas = np.zeros((pad_h, pad_w, 3), dtype=img_norm.dtype)
        canvas[:tgt_h, :tgt_w, :] = img_norm
        img_norm = canvas

    # === 5) HWC -> NCHW, 放回容器 ===
    img_t = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).to(device)
    maybe_tensor = set_img(sample_or_data['img'], img_t)
    if maybe_tensor is not None:
        sample_or_data['img'] = [maybe_tensor]


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    test_adv_cfg=None,
                    adv_load_dir="/root/anaconda3/oddefense/datasets/adv_voc"):

    
    if test_adv_cfg is not None:
        adv_flag = test_adv_cfg.get("adv_flag", False)
    else:
        adv_flag = False
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if adv_flag and (adv_load_dir is None):
            sample = scatter(data, [torch.cuda.current_device()])[0]
            img = sample['img']

            img_mean = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['mean']).to(img.device)
            img_mean = img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            img_std = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['std']).to(img.device)
            img_std = img_std.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            

            img_transform = (lambda x: (x - img_mean) / img_std, lambda x: x * img_std + img_mean)
            
            img = img_transform[1](img)
            # adv_sample = copy.deepcopy(sample)
            adv_sample = sample
            img_adv = cal_adv(model, img, adv_sample, img_transform, test_adv_cfg)
            
            # 取出保存配置
            save_adv = bool(test_adv_cfg.get("save_adv", False)) if test_adv_cfg else False
            save_dir = test_adv_cfg.get("save_dir", None) if test_adv_cfg else None
            save_ext = test_adv_cfg.get("save_ext", ".png") if test_adv_cfg else ".png"

            img_adv = img_transform[0](img_adv)

            if save_adv and save_dir is not None:
                # 注意：此处需要用“原图文件名”
                ori_name = sample['img_metas'][0]["ori_filename"]
                _save_adv_image(img_adv, img_transform, ori_name, save_dir, save_ext=save_ext)

            # print(torch.max(torch.abs(img_adv - sample['img'])))

            sample.pop('gt_bboxes')
            sample.pop('gt_labels')
            if 'gt_masks' in sample:
                sample.pop('gt_masks')
            if 'gt_semantic_seg' in sample:
                sample.pop('gt_semantic_seg')
            sample['img_metas'] = [sample['img_metas']]
            sample['img'] = [img_adv.detach()]
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **sample)
        elif adv_load_dir is not None:
            # 直接从磁盘读取对抗图替换后推理
            data.pop('gt_bboxes'); data.pop('gt_labels')
            if 'gt_masks' in data: data.pop('gt_masks')
            if 'gt_semantic_seg' in data: data.pop('gt_semantic_seg')
            data['img_metas'] = [data['img_metas']]
            data['img'] = [data['img']]
            apply_saved_adv_to_sample(data, adv_load_dir, torch.cuda.current_device())
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
        else:
            data.pop('gt_bboxes')
            data.pop('gt_labels')
            if 'gt_masks' in data:
                data.pop('gt_masks')
            if 'gt_semantic_seg' in data:
                data.pop('gt_semantic_seg')
            data['img_metas'] = [data['img_metas']]
            data['img'] = [data['img']]
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, test_adv_cfg=None, adv_load_dir="/root/anaconda3/oddefense/black_coco_datasets/adv_coco_fcos_r50_cls"):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """

    if test_adv_cfg is not None:
        adv_flag = test_adv_cfg.get("adv_flag", False)
    else:
        adv_flag = False

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        if adv_flag and (adv_load_dir is None):
            sample = scatter(data, [torch.cuda.current_device()])[0]
            img = sample['img']

            img_mean = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['mean']).to(img.device)
            img_mean = img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            img_std = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['std']).to(img.device)
            img_std = img_std.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            

            img_transform = (lambda x: (x - img_mean) / img_std, lambda x: x * img_std + img_mean)
            
            img = img_transform[1](img)
            # adv_sample = copy.deepcopy(sample)
            adv_sample = sample
            img_adv = cal_adv(model, img, adv_sample, img_transform, test_adv_cfg)
            
            # 取出保存配置
            save_adv = bool(test_adv_cfg.get("save_adv", False)) if test_adv_cfg else False
            save_dir = test_adv_cfg.get("save_dir", None) if test_adv_cfg else None
            save_ext = test_adv_cfg.get("save_ext", ".png") if test_adv_cfg else ".png"

            img_adv = img_transform[0](img_adv)

            if save_adv and (save_dir is not None):
                # 注意：此处需要用“原图文件名”
                ori_name = sample['img_metas'][0]["ori_filename"]
                _save_adv_image(img_adv, img_transform, ori_name, save_dir, save_ext=save_ext)

            # print(torch.max(torch.abs(img_adv - sample['img'])))

            sample.pop('gt_bboxes')
            sample.pop('gt_labels')
            if 'gt_masks' in sample:
                sample.pop('gt_masks')
            if 'gt_semantic_seg' in sample:
                sample.pop('gt_semantic_seg')

            # save = True
            # if save:
            #     root = "/home/lixiao/ssd/workdir/oddefense/dataset/" + "dndetr_resnet_all"
            #     if not os.path.exists(root):
            #         os.mkdir(root)
            #     to_save = img_transform[1](img_adv)
            #     name = sample['img_metas'][0]["ori_filename"]
            #     cv2.imwrite(os.path.join(root, name), to_save.squeeze(0).permute(1,2,0).detach().cpu().numpy()[:,:,(2,1,0)])

            # root = "/home/lixiao/ssd/workdir/oddefense/dataset/" + "pascal_new/"
            # if not os.path.exists(root):
            #     os.mkdir(root)
            # to_save = img_transform[1](img_adv)
            # name = sample['img_metas'][0]["ori_filename"]
            # cv2.imwrite(os.path.join(root, name.split(".")[0] + ".png"), to_save.squeeze(0).permute(1,2,0).detach().cpu().numpy()[:,:,(2,1,0)])

            sample['img_metas'] = [sample['img_metas']]
            sample['img'] = [img_adv.detach()]
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **sample)
        elif adv_load_dir is not None:
            data.pop('gt_bboxes'); data.pop('gt_labels')
            if 'gt_masks' in data: data.pop('gt_masks')
            if 'gt_semantic_seg' in data: data.pop('gt_semantic_seg')
            data['img_metas'] = [data['img_metas']]
            data['img'] = [data['img']]
            apply_saved_adv_to_sample(data, adv_load_dir, torch.cuda.current_device())
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
        else:
            data.pop('gt_bboxes')
            data.pop('gt_labels')
            if 'gt_masks' in data:
                data.pop('gt_masks')
            if 'gt_semantic_seg' in data:
                data.pop('gt_semantic_seg')
            
            # alter = True
            # if alter:
            #     folders = ["none", "frcnn_resnet_all", "fcos_resnet_all", "dndetr_resnet_all", "frcnn_conv_all", "fcos_conv_all", "dndetr_conv_all"]
            #     root = "/home/lixiao/data3/workdir/oddefense/dataset/" + folders[6]
            #     name = data['img_metas']._data[0][0]["ori_filename"]
            #     newimg = cv2.imread(os.path.join(root, name), cv2.COLOR_BGR2RGB)
            #     newimg = torch.from_numpy(newimg).permute(2, 0, 1).unsqueeze(0).float()
            #     # newimg = img_transform[1](newimg)
            #     newimg[0, 0, :, :] = (newimg[0, 0, :, :] - 123.675) / 58.395
            #     newimg[0, 1, :, :] = (newimg[0, 1, :, :] - 116.28) / 57.12
            #     newimg[0, 2, :, :] = (newimg[0, 2, :, :] - 103.53) / 57.375
            #     data['img']._data[0] = newimg.to(data['img']._data[0].device)
                


            data['img_metas'] = [data['img_metas']]
            data['img'] = [data['img']]
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
        

            # encode mask results
        with torch.no_grad():
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                            for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
