import torch

Params = {
    1: dict(
        exp_id =  "0_resnet",
        data_root =  "", # folder containing .rec file
        save_dir =  "", # models are saved here
        multi_flag =  0, # set to 1 to distribute loss layer weights among mutiple GPUs
        batch_size =  64,
        epochs =  67,
        lr = 0.1,
        moment =  0.9,
        decay =  0.0005,
        model_name =  "resnet50",
        r =  1.0,
        delta =  0.5,
        top_k =  1.0,

    ),
}

