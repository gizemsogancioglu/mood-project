# Explainable apparent personality prediction over mood states
Here, we provide source code and the annotated dataset for the following paper: [Can mood primitives predict apparent personality?, Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]

Available in this repository: 
- Annotated mood dataset (data/mood_annots.csv)  (960 videos which are subset of ChaLearn LAP-FI Dataset)
- Extracted features from the videos of ChaLearn LAP-FI dataset (features/feature_[train/test/validation].csv)  
- Scripts for mood and personality prediction (source/*.py)  

        .
        ├── source                          # including source files (feature extractor, training, and predictor scripts)                
        │   ├── mood_model.py               # main script for valence/arousal/likeability classifier
        │   ├── personality_predictor.py    # main script for personality regressor using predicted mood dimensions
        │   ├── feature_extractor           # a folder containing feature extractor scripts
        ├── data                         
        │   ├── mood_annots.py              # contains valence, arousal, likeability annotations 
        ├── experiments                 
        │   ├── ff_ling_exp.csv             # feature-fusion experiments among a set of linguistic features
        │   ├── multi-modal_exp.csv         # feature-fusion multi-modal experiments
        └── features                        # extracted linguistic, acoustic, visual features. If this folder is not empty, the feature extraction step will be skipped. 

## Pipeline of Apparent Personality Predictor

![Alt text](pipeline.png?raw=true "The proposed apparent personality prediction model")

- Steps of the algorithm as follows:
    * Reading annotations 
    * Feature extraction (If feature files are in the features folder, the feature extraction step will be skipped.)
    * Mood Classification (If predictions folder contains mood predictions already, the mood classification step will be skipped.)
    * Personality trait regressor

## How to run?
<strong> Run the following command to start docker container. Predictions will be saved in /tmp folder. </strong> 

```bash
docker run gizemsogancioglu/mood_img:latest 
```

> `personality_predictor.py` is the main script for the prediction of personality scores from the videos of people. `IMPORTANT NOTE`: 
    annotation_{test/training/validation}.pkl files should be in data/p_traits folder to run this script. (Due to confidentiality, we can not upload them. Please contact CVPR LAP-FI Challenge organizers to have access to this dataset.)


## TODO
We are working on the bug in the mood classification module.
That´s why, in the case of an empty predictions folder, predicted results might be wrong. Sorry for the inconvenience.

## References
* Paper: [Can mood primitives predict apparent personality?, Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]
* For more information or any problems, please contact: gizemsogancioglu@gmail.com
