# Possible Improvements:

- [ ] Improve Face Tracking Pipeline:
  - [X] Face Detection (RetinaFace): `SATISFIED, Improve if a better model is found`
  - [ ] Face Alignment:
    - [ ] Update core landmark detection (MTCNN -> ___)
    - [ ] Update face alignment model ([Default face.evoLVE](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/applications/align/face_align.py) - > [Insight Face's](https://github.com/deepinsight/insightface/tree/master/alignment/heatmap))
  - [ ] Face Recognition/Verification: 
    - [ ] Better implementation of Arcface ([face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/util/extract_feature_v1.py) - > [InsightFace's](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch))
