# Explainable apparent personality prediction over mood states
Here, we provide source code and the annotated dataset for the following paper: [Can mood primitives predict apperant personality?, Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]

- Annotated mood dataset (data/mood_annots.csv)  (960 videos which are subset of ChaLearn LAP-FI Challenge Dataset)
- Extracted features (features/feature_[train/test/validation].csv)  
- Scripts for mood and personality prediction (source/*.py)  

## Feature Extractor


## Mood Classifier


## Apparent Personality Predictor

![Alt text](pipeline.png?raw=true "The proposed apparent personality prediction model")

    .
    ├── source                      # including source files (feature extractor, training and predictor scripts)                
    │   ├── mood_model.py           # Valence/arousal/likeability classifier
    │   ├── feature_extractor       # a folder containing feature extractor scripts
    ├── data                         
    │   ├── mood_annots.py          # contains valence, arousal, likeability annotations 
    ├── experiments                 
    │   ├── ff_ling_exp.csv         # feature-fusion experiments among a set of linguistic features
    │   ├── multi-modal_exp.csv     # feature-fusion multi-modal experiments
    └── features                    # Extracted linguistic, acoustic, visual features. If this folder is not empty, feature extraction step will be skipped. 
    

## References
* Paper: [Can mood primitives predict apperant personality?, Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]
* For more information or any problems, please contact: gizemsogancioglu@gmail.com
