# RST
Single Domain Generalized Fundus Image Segmentation via Random Style Transformation
Please read our [paper](https://doi.org/xxx) for more details!

### Introduction:
Accurate segmentation of fundus structures plays a vital role in the diagnosis of ophthalmic diseases. Recently, deep learning methods have achieved promising results on fundus image segmentation. However, they suffer from the domain shift problem, i.e., the segmentation performance drops significantly when the test images are sampled from a distribution distinct from that of the training images. To overcome this challenge, this paper focuses on the challenging single source domain generalization, where we aim to train a deep learning model on only one source training domain and expect it to perform well on other unseen target domains.  Our contributions are threefold. First, we propose Random Style Transformation (RST), a novel approach for single-source domain generalized fundus image segmentation. Second, RST enhances image diversity by leveraging a reconstruction network and applying style transformation to its intermediate features, generating images with varied colors and styles. Third, we comprehensively evaluate the effectiveness of our method on four fundus image segmentation tasks under both single- and multi-domain generalization settings, demonstrating that RST consistently outperforms several state-of-the-art comparison methods.

## Training Pipeline
We provide a script to training on the DRIVE dataset, just use the following command:
```bash
python3 train_recnet.py > log_rec_drive 2>&1
```
After training the RecNet, run the following command to train the segmentation model:
```bash
python3 train_seg.py > log_seg_drive 2>&1
```
After training, the model checkpoints will be saved in the `snapshot/` directory.

### 📦 Pretrained Models
We also provide several pre-trained checkpoints for DRIVE dataset in the 'snapshot' folder.

## 🔍 Testing
Once the training is complete, you can test the model on a different dataset using the following command:

```bash
python3 predict_seg.py drive stare
```

```bash
python3 predict_seg.py drive chase
```

```bash
python3 predict_seg.py drive rc-slo
```
