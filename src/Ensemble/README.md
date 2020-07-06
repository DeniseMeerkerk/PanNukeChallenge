# Ensemble

## Files
* `ensemble_pipeline_split{1|2|3}.py`
  Using the results of Hover-Net and Micro-Net it outputs mask-like images for the softvote. It calculates the Jaccard score (sklearn) by comparing to the ground truth for the softvote and hardvote ensemble as well as the original methods HoVer-Net and Micro-Net, which is saved in a `.csv` file.
* `fold3_hardvote.csv` scores for the hardvote ensemble (example for split 1)
* `fold3_softvote.csv` scores for the softvote ensemble (example for split 1)

## Images
![](https://github.com/DeniseMeerkerk/PanNukeChallenge/blob/master/src/Ensemble/example_score_weight.png)

From this image is clearly visible that a combination of two methods can improve the scoring, in this case jaccard score.

![https://docs.google.com/spreadsheets/d/1j9-C2HHMc7Z9PQr_gq5FS9y6Ing7vT0fEKp59f8X3Jo/edit?usp=sharing](https://github.com/DeniseMeerkerk/PanNukeChallenge/blob/master/src/Ensemble/chart.png)

To see how this chart is made go to [this google sheet](https://docs.google.com/spreadsheets/d/1j9-C2HHMc7Z9PQr_gq5FS9y6Ing7vT0fEKp59f8X3Jo/edit?usp=sharing). Our ensemble methods outperform each of the original models (Hover-Net and Micro-Net). Surprising is that hardvoting performs better than trained softvoting. We think this is due to the fact that the softvote weight is trained on a balanced (on tissue types) validation subset, which is not representative of the whole dataset.


![](https://github.com/DeniseMeerkerk/PanNukeChallenge/blob/master/src/Ensemble/exampleresult2.png)

The colors of this picture can not be directly compared to each other concerning the classes. It is slightly different when the prediction contains a different number of classes, in this case from left to right 3, 4, 3 ,4 ,3 classes.
