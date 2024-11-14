
# topics:

## **DFR**
Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations \
https://arxiv.org/abs/2204.02937
https://github.com/PolinaKirichenko/deep_feature_reweighting

To train the DFR, run the following command:

```bash
python dfr_evaluate_spurious.py \
  --data_dir /scratch_shared/primmere/isic/isic_224/raw_224_with_selected \
  --result_path /home/primmere/logs/result_isic_2.pkl \
  --ckpt_path /home/primmere/logs/isic_logs_2/vit_isic_final_checkpoint_test.pt \
  --skip_dfr_train_subset_tune True \
  --model_type "vit_b_16" \
  --batch_size 32 \
  --seed 7 \
  --DFR_retrained_model_path /home/primmere/logs/isic_logs_2/dfr_isic_model.pth \
  --DFR_logreg_path /home/primmere/logs/isic_logs_2/dfr_isic_logreg_2.pth \
  --label_csv /scratch_shared/primmere/isic/metadata_w_split.csv \
  --dataset "isic"
```

#### **Outputs**
- *Result:* `.pkl` file accuracies of different models.
- *DFR_retrained_model:* `.pth` file of the model with the last layer retrained.
- *DFR_logreg:* `.pth` file for the logistic regression model (only the last layer).

---

## **Training an ISIC ViT**
To train a Vision Transformer on the ISIC dataset:

```bash
python train_classifier_isic.py \
  --data_dir "/scratch_shared/primmere/isic/isic_224/raw_224" \
  --output_dir "/home/primmere/logs/isic_logs" \
  --eval_freq 10 \
  --seed 7 \
  --label_csv "/scratch_shared/primmere/isic/metadata.csv" \
  --num_epochs 200 \
  --pre_split True
```
#### **Outputs**
- *Model:* `.pt` file with the model, saved in output_dir

---

## **Transformer Visualization with LRP**
Transformer Interpretability Beyond Attention Visualization \
https://arxiv.org/abs/2012.09838 \
https://github.com/hila-chefer/Transformer-Explainability \

To generate visualizations with LRP:

- Example 1:

```bash
python dfr_transformer_explainability.py \
  --img_path catdog.png \
  --chkpt_path ../dfr/logs/dfr_model.pth \
  --output_dir visualisations \
  --output_filename_prefix test
```

- Example 2:

```bash
python dfr_transformer_explainability.py \
  --chkpt_path /home/primmere/logs/isic_logs_3/vit_isic_final_checkpoint_test.pt \
  --output_dir /home/primmere/outputs/isic_visualisations \
  --img_path /scratch_shared/primmere/isic/isic_224/raw_224_with_selected/ISIC_0055226.jpg
```

---

## **Extras**

Download models from the cluster
```bash
scp -P 223 -r primmere@marc3a.hrz.uni-marburg.de:/home/primmere/logs/isic_logs_3/dfr_isic_model.pth \
C:/Users/elmop/deep_feature_reweighting/deep_feature_reweighting/ISIC_ViT/

scp -P 223 -r primmere@marc3a.hrz.uni-marburg.de:/home/primmere/logs/isic_logs_3/vit_isic_final_checkpoint_test.pt \
C:/Users/elmop/deep_feature_reweighting/deep_feature_reweighting/ISIC_ViT/
```
