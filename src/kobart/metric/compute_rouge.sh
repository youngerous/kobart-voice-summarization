# 모든 checkpoint에 대한 rouge score 계산

python get_rouge.py --ckpt  "./ckpt/best_model_step_20365_loss_1.9004.pt" --generation False
python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8739.pt" --generation False
python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8838.pt" --generation False
python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8953.pt" --generation False


#python get_rouge.py --ckpt  "./ckpt/best_model_step_20365_loss_1.9004.pt"
#python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8739.pt"
#python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8838.pt"
#python get_rouge.py --ckpt  "./ckpt/best_model_step_40730_loss_1.8953.pt"
