import os
import torch
import torchvision
import cv2
import numpy as np
import random

# IQT
from model.model_main import IQARegression
from model.backbone import inceptionresnetv2, Mixed_5b, Block35, SaveOutput
from config.config import config_all
# DISTS
from DISTS_pt import DISTS, L2pooling, prepare_image
from PIL import Image
# FID
from fid_score import save_fid_stats, calculate_fid_given_paths
# KID
from kid_score import calculate_kid_given_paths
# lpips, psnr, ms-ssim
from lpips_score import compute_metrics
from loss import perceptual_loss as ps

def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = False
    return seed 


def is_png(filename):
    ext = os.path.splitext(filename)[-1]
    return ext == '.png'

def iqt_init():
    config = config_all()
    config.device = torch.device("cuda:%s" %config.GPU_ID if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU %s' % config.GPU_ID)
    else:
        print('Using CPU')

    # create_model
    model_transformer = IQARegression(config).to(config.device)
    model_backbone = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background').to(config.device)
    # save intermediate layers
    save_output = SaveOutput()
    hook_handles = []
    for layer in model_backbone.modules():
        if isinstance(layer, Mixed_5b):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)
        elif isinstance(layer, Block35):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)

    # load weights
    if config.weight_file is not None:
        checkpoint = torch.load(config.weight_file,map_location='cuda:0')
        model_transformer.load_state_dict(checkpoint['model_state_dict'])

        model_transformer.eval()
        model_backbone.eval()
    else:
        raise ValueError('You need to specify a weight file.')
    if not os.path.exists(config.exp_path ):
        os.makedirs(config.exp_path, exist_ok = True)
    return config, model_transformer, model_backbone, save_output

def iqt_calculate(config, model_transformer, model_backbone, save_output ):
    # test images
    filenames = os.listdir(config.ori_path)
    filenames.sort()
    f = open(config.exp_path +  config.result_file, 'w')

    sums = 0.0

    line = "IQT\n"
    f.write(line)

    for filename in filenames:
        d_img_name = os.path.join(config.exp_path, filename)
        ext = os.path.splitext(d_img_name)[-1]
        
        enc_inputs = torch.ones(1, config.n_enc_seq+1).to(config.device)
        dec_inputs = torch.ones(1, config.n_dec_seq+1).to(config.device)
        if ext == '.png':
            # reference image
            r_img_name = filename[:-4] + '.png'
            r_img = cv2.imread(os.path.join(config.ori_path, r_img_name), cv2.IMREAD_COLOR)
            r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
            r_img = np.array(r_img).astype('float32') / 255
            r_img = (r_img - 0.5) / 0.5
            r_img = np.transpose(r_img, (2, 0, 1))
            r_img = torch.from_numpy(r_img)
            
            # distoted image
            # print(config.ori_path, d_img_name)
            d_img = cv2.imread( d_img_name, cv2.IMREAD_COLOR)
            d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
            d_img = np.array(d_img).astype('float32') / 255
            d_img = (d_img - 0.5) / 0.5
            d_img = np.transpose(d_img, (2, 0, 1))
            d_img = torch.from_numpy(d_img)

            pred = 0.0
            # inference (use ensemble or not)
            if config.test_ensemble:
                for i in range(config.n_ensemble):
                    c, h, w = r_img.size()
                    new_h = config.crop_size
                    new_w = config.crop_size
                    top = np.random.randint(0, h - new_h)
                    left = np.random.randint(0, w - new_w)

                    r_img_crop = r_img[:, top: top+new_h, left: left+new_w].unsqueeze(0)
                    d_img_crop = d_img[:, top: top+new_h, left: left+new_w].unsqueeze(0)

                    r_img_crop = r_img_crop.to(config.device)
                    d_img_crop = d_img_crop.to(config.device)

                    # backbone feature map (ref)
                    x_ref = model_backbone(r_img_crop)
                    feat_ref = torch.cat(
                        (save_output.outputs[0],
                        save_output.outputs[2],
                        save_output.outputs[4],
                        save_output.outputs[6],
                        save_output.outputs[8],
                        save_output.outputs[10]),
                        dim=1
                    ) # feat_ref: n_batch x (320*6) x 21 x 21
                    # clear list (for saving feature map of d_img)
                    save_output.outputs.clear()

                    # backbone feature map (dis)
                    x_dis = model_backbone(d_img_crop)
                    feat_dis = torch.cat(
                        (save_output.outputs[0],
                        save_output.outputs[2],
                        save_output.outputs[4],
                        save_output.outputs[6],
                        save_output.outputs[8],
                        save_output.outputs[10]),
                        dim=1
                    ) # feat_ref: n_batch x (320*6) x 21 x 21
                    # clear list (for saving feature map of r_img in next iteration)
                    save_output.outputs.clear()

                    feat_diff = feat_ref - feat_dis
                    enc_inputs_embed = feat_diff
                    dec_inputs_embed = feat_ref
                    pred += model_transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)
                    
                pred /= config.n_ensemble
                
            else:
                c, h, w = r_img.size()
                new_h = config.crop_size
                new_w = config.crop_size
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)

                r_img_crop = r_img[:, top: top+new_h, left: left+new_w].unsqueeze(0)
                d_img_crop = d_img[:, top: top+new_h, left: left+new_w].unsqueeze(0)

                r_img_crop = r_img_crop.to(config.device)
                d_img_crop = d_img_crop.to(config.device)

                # backbone feature map (ref)
                x_ref = model_backbone(r_img_crop)
                feat_ref = torch.cat(
                    (save_output.outputs[0],
                    save_output.outputs[2],
                    save_output.outputs[4],
                    save_output.outputs[6],
                    save_output.outputs[8],
                    save_output.outputs[10]),
                    dim=1
                ) # feat_ref: n_batch x (320*6) x 21 x 21
                # clear list (for saving feature map of d_img)
                save_output.outputs.clear()

                # backbone feature map (dis)
                x_dis = model_backbone(d_img_crop)
                feat_dis = torch.cat(
                    (save_output.outputs[0],
                    save_output.outputs[2],
                    save_output.outputs[4],
                    save_output.outputs[6],
                    save_output.outputs[8],
                    save_output.outputs[10]),
                    dim=1
                ) # feat_ref: n_batch x (320*6) x 21 x 21
                # clear list (for saving feature map of r_img in next iteration)
                save_output.outputs.clear()

                feat_diff = feat_ref - feat_dis
                enc_inputs_embed = feat_diff
                dec_inputs_embed = feat_ref

                pred = model_transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)

            sums += float(pred.item())
            line = "%s,%f\n" % (filename, float(pred.item()))
            f.write(line)
    ave = (sums/len(filenames))
    line = "IQT ave,%f\n" % (ave)
    f.write(line)
    f.close()
    return ave

def dist_calculate(config):
    # prepare file
    filenames = os.listdir(config.ori_path)
    filenames.sort()
    f = open(config.exp_path +  config.result_file, 'w')
    sums = 0
    line = "DISTS\n"
    f.write(line)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = DISTS().to(device)

    for filename in filenames:
        dist = prepare_image(Image.open(os.path.join(config.exp_path, filename)))
        ref = prepare_image(Image.open(os.path.join(config.ori_path, filename)))
        ref = ref.to(device)
        dist = dist.to(device)
        score = model(ref, dist)
        sums += score.item()
        # print(score.item())
        line = "%s,%f\n" % (filename, float(score.item()))
        f.write(line)
    print(sums/len(filenames))
    ave = float(sums/len(filenames))
    line = "DISTS ave,%f\n" % (ave)
    f.write(line)
    f.close()
    return ave

def fkid_calculate(config):
    path = [config.ori_path, config.exp_path]
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
    num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    

    fid_value = calculate_fid_given_paths(path,
                                          config.batch_size,
                                          config.device,
                                          config.dims,
                                          num_workers)
    print('FID: ', fid_value)
    kid_results = calculate_kid_given_paths(path, config.batch_size, config.device == 'cuda', config.dims, model_type='inception')
    kid_value = 0.0
    for p, m, s in kid_results:
        print('KID (%s): %.5f (%.5f)' % (p, m, s))
        kid_value = m
    f = open(config.exp_path +  config.result_file, 'w')
    line = "FID:%f\n" % (fid_value)
    line = "KID:%f\n" % (kid_value)

    f.write(line)
    f.close()
    return fid_value, kid_value

def lpips_calculate(config):
    # prepare file
    filenames = os.listdir(config.ori_path)
    filenames.sort()
    f = open(config.exp_path +  config.result_file, 'w')
    sums = {
        "lpips":0.0,
        "psnr":0.0,
        "ms-ssim":0.0

    }
    line = "LPIPS\n"
    f.write(line)
    device = config.device
    lpips = ps.PerceptualLoss(model='net-lin', net='vgg',
                               use_gpu=torch.cuda.is_available(),gpu_ids=0)
    lens = len(filenames)

    for filename in filenames:
        img_dist = Image.open(os.path.join(config.exp_path, filename))
        img_ref = Image.open(os.path.join(config.ori_path, filename))
        psnr, ssim = compute_metrics(img_dist, img_ref)
        sums["psnr"] += psnr
        sums["ms-ssim"] += ssim
        dist = prepare_image(img_dist)
        ref = prepare_image(img_ref)
        ref = ref.to(device)
        dist = dist.to(device)
        lpips_score = lpips(ref, dist)
        sums["lpips"] += lpips_score.item()
        # print(score.item())
        line = "%s,%f, %f, %f\n" % (filename, sums["lpips"]/lens, sums["psnr"]/lens, sums["ms-ssim"]/lens)
        f.write(line)

    line = "LPIPS ave: %f, PSNR ave:%f, MS-SSIM ave:%f\n" % (sums["lpips"]/lens, sums["psnr"]/lens, sums["ms-ssim"]/lens)
    print("LPIPS ave: %f, PSNR ave:%f, MS-SSIM ave:%f\n" % (sums["lpips"]/lens, sums["psnr"]/lens, sums["ms-ssim"]/lens))
    f.write(line)
    f.close()
    return sums["lpips"]/lens, sums["psnr"]/lens, sums["ms-ssim"]/lens

def main():
    setup_seed()
    config_all, model_transformer, model_backbone, save_output = iqt_init()
    
    iqt = iqt_calculate(config_all, model_transformer, model_backbone, save_output )
    dist = dist_calculate(config_all)
    fid, kid = fkid_calculate(config_all)
    lpips, psnr, ssim = lpips_calculate(config_all)
    print(config_all.exp_path)
    print("iqt:%f \n dist:%f \n fid:%f \n kid:%f \n lpips:%f \n psnr:%f \n ms-ssim:%f \n"%(iqt,dist,fid,kid,lpips,psnr,ssim))

if __name__ == '__main__':
    main()
