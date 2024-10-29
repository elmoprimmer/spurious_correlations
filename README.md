**Train dfr on model:** \
python3 dfr_evaluate_spurious.py --data_dir=/scratch_shared/primmere/isic/isic_224/raw_224_with_selected --result_path=/home/primmere/logs/result_isic.pkl --ckpt_path=/home/primmere/logs/isic_logs/vit_isic_final_checkpoint.pt --skip_dfr_train_subset_tune=True --model_type="vit_b_16" --batch_size=32 --seed=7 --DFR_retrained_model_path=/home/primmere/logs/isic_logs/dfr_isic_model.pth --DFR_logreg_path=/home/primmere/logs/isic_logs/dfr_isic_logreg.pth --label_csv=/scratch_shared/primmere/isic/metadata_w_insert.csv --dataset="isic"


Result = pkl of logreg performance  
DFR_retrained = .pth of new model with last layer retraining
DFR_logreg = .pth of logreg model (only last layer)


**Train ISIC ViT**
python train_classifier_isic.py --data_dir "/scratch_shared/primmere/isic/isic_224/raw_224" --output_dir "/home/primmere/logs/isic_logs" --eval_freq 2 --seed 7 --label_csv "/scratch_shared/primmere/isic/metadata.csv" --num_epochs 2

These files are in external/dfr/


**Transformer Visualisation**
python dfr_transformer_explainability.py --img_path catdog.png --chkpt_path ../dfr/logs/dfr_model.pth --output_dir visualisations --output_filename_prefix test