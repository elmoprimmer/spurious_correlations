**Train dfr on model:** \
python dfr_evaluate_spurious.py --data_dir=/scratch_shared/primmere/isic/isic_224/raw_224_with_selected --result_path=/home/primmere/logs/result_isic_2.pkl --ckpt_path=/home/primmere/logs/isic_logs_2/vit_isic_final_checkpoint_test.pt --skip_dfr_train_subset_tune=True --model_type="vit_b_16" --batch_size=32 --seed=7 --DFR_retrained_model_path=/home/primmere/logs/isic_logs_2/dfr_isic_model.pth --DFR_logreg_path=/home/primmere/logs/isic_logs_2/dfr_isic_logreg_2.pth --label_csv=/scratch_shared/primmere/isic/metadata_w_split.csv --dataset="isic"


Result = pkl of logreg performance  
DFR_retrained = .pth of new model with last layer retraining
DFR_logreg = .pth of logreg model (only last layer)


**Train ISIC ViT:** \
python train_classifier_isic.py --data_dir "/scratch_shared/primmere/isic/isic_224/raw_224" --output_dir "/home/primmere/logs/isic_logs" --eval_freq 2 --seed 7 --label_csv "/scratch_shared/primmere/isic/metadata.csv" --num_epochs 2 --pre_split True

These files are in external/dfr/


**Transformer Visualisation:** \
python dfr_transformer_explainability.py --img_path catdog.png --chkpt_path ../dfr/logs/dfr_model.pth --output_dir visualisations --output_filename_prefix test
