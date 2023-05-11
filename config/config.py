from option.config import Config

def config_all():
    config = Config({
    # device
    "GPU_ID": "0",
    
    # model for PIPAL (NTIRE2021 Challenge)
    "n_enc_seq": 21*21,                 # feature map dimension (H x W) from backbone, this size is related to crop_size
    "n_dec_seq": 21*21,                 # feature map dimension (H x W) from backbone
    "n_layer": 1,                       # number of encoder/decoder layers
    "d_hidn": 128,                      # input channel (C) of encoder / decoder (input: C x N)
    "i_pad": 0,
    "d_ff": 1024,                       # feed forward hidden layer dimension
    "d_MLP_head": 128,                  # hidden layer of final MLP 
    "n_head": 4,                        # number of head (in multi-head attention)
    "d_head": 128,                      # input channel (C) of each head (input: C x N) -> same as d_hidn
    "dropout": 0.1,                     # dropout ratio of transformer
    "emb_dropout": 0.1,                 # dropout ratio of input embedding
    "layer_norm_epsilon": 1e-12,
    "n_output": 1,                      # dimension of final prediction
    "crop_size": 192,                   # input image crop size

    # data

    #"ori_path":"/dockerdata/peirongning/workspace/result/kodak/ori_gt/",
    #"exp_path":"/dockerdata/peirongning/workspace/result/kodak/poelic_bpp5359/",
    #"ori_path": "/dockerdata/peirongning/dataset/imagedataset/professional_test_2021/",
    #"exp_path":"/dockerdata/peirongning/workspace/result/HIFIC/clic20test_high/",
    #"ori_path":"/dockerdata/peirongning/workspace/result/clic20test/ori/",
    #"exp_path":"/dockerdata/peirongning/workspace/result/clic20test/poelic_med/",
    "ori_path":"/dockerdata/peirongning/dataset/imagedataset/clic20test/",
    "exp_path":"/dockerdata/peirongning/workspace/result/HIFIC/clic20test_low/",
    #"ori_path":"/dockerdata/peirongning/workspace/result/clic22val/ori_gt/",
    #"exp_path":"/dockerdata/peirongning/workspace/result/clic22val/poelic_bpp1460/",
    #"ori_path":"/dockerdata/peirongning/workspace/result/CLIC2020Professional_test/ori",
    #"exp_path":"/dockerdata/peirongning/workspace/result/CLIC2020Professional_test/ourbpp02275/801/",
    "weight_file": "./weights/PIPAL/epoch40.pth", # "./weights/epoch240.pth",
    "result_file":   "output.txt",

    # ensemble in test
    "test_ensemble": True,
    "n_ensemble": 20,
    
    # FID
    "save_stats": False,
    "dims":2048,
    "batch_size":1
    })
    return config
