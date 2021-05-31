# 모든 checkpoint에 대한 rouge score 계산

python get_rouge.py --ckpt  "./ckpt/best_model_step_20365_loss_1.9651.pt" --generation False
python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8739.pt" --generation False
python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8838.pt" --generation False
python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8953.pt" --generation False


#python get_rouge.py --ckpt  "./ckpt/best_model_step_20365_loss_1.9004.pt"
#python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8739.pt"
#python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8838.pt"
#python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8953.pt"

# DistilBART rouge score 계산

# N_ENC=3
# N_DEC=6
# python get_rouge.py --ckpt  "./ckpt/best_model_step_48876_loss_1.8759.pt" --generation True --n_enc=${N_ENC} --n_dec=${N_DEC}


# N_ENC=3
# N_DEC=3
# python get_rouge.py --ckpt  "./ckpt/best_model_step_65168_loss_2.1756.pt" --generation False --n_enc=${N_ENC} --n_dec=${N_DEC}

# N_ENC=6
# N_DEC=3
# python get_rouge.py --ckpt  "./ckpt/best_model_step_65168_loss_2.2167.pt" --generation False --n_enc=${N_ENC} --n_dec=${N_DEC}

