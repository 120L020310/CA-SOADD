# CA-SOADD
Code for "Centroid-Anchored Boundary Shaping for Strict One-Class Audio Deepfake Detection"


## Abstract

The rapid evolution of audio deepfakes necessitates robust detection systems capable of generalizing to unseen attacks. While strict real-only one-class learning offers inherent robustness for this task, it faces the central challenge of defining tight decision boundaries without utilizing spoof data for training or parameter tuning. Existing ``relaxed" approaches often compromise this strictness by introducing auxiliary negatives, resulting in boundaries biased toward seen artifacts and poor generalization to unseen attacks. We address this by proposing CA-SOADD, a framework that constructs off-manifold boundary probes to explicitly shape the acceptance region. Our centroid-anchored tri-objective learning paradigm enforces centroid compactness and a centroid-referenced margin against probes, without treating them as an explicit negative class. Furthermore, we extend the framework to heterogeneous settings via domain-conditioned centroids. Experiments on ASVSpoof and MLAAD demonstrate that our strict real-only method consistently outperforms baselines under unseen attacks and domain shifts, validated by extensive ablations.


## Usage


One can run the following commands to train or test our model.
### Train
```bash
python train_CA_SOADD.py --gpu 0 --cfg 'ASV2021_LA'  -v 0;\

python train_cross_domain.py --gpu 0 --cfg 'MLAAD_same_lang'  -v 0;\
```
### Test
```bash
python train_CA_SOADD.py --gpu 0 --cfg 'ASV2021_LA'  -v 0 -t 1;\

python train_cross_domain.py --gpu 0 --cfg 'MLAAD_same_lang'  -v 0 -t 1;\
```

## Two Stage Train
If you wish to observe the degree of compactness, you can set --metric eer and warm_epoch -1 to train compactness independently:

```bash
python train_CA_SOADD.py --gpu 0 --cfg 'ASV2021_LA'  -v 0 --metric eer --warm_epoch -1;\

```

After obtaining and saving a suitable checkpoint, further set warm_epoch=0 to continue the subsequent training:
```bash
python train_CA_SOADD.py --gpu 0 --cfg 'ASV2021_LA'  -v 0 --resume 1 --metric loss --warm_epoch 0;\
```