## TrackRec: Iterative Alternating Feedback with Chain-of-Thought via Preference Alignment for Recommendation

## Setup

1. Download dataset
   
   Take Amazon-Books for example, download the dataset to folder `data/amz/proc_data/` on [Here]([https://drive.google.com/drive/folders/1OdL6JPq_UZUSCO3skAIX3NOxF81goB3F?usp=sharing](https://drive.google.com/drive/folders/1hZHRhdNC9espzom_ySpXH7EA78njZfci?usp=drive_link))

2. We have provided our processed dataset as well as the distilled dataset. We have retained a small portion of the data in the code, and the complete dataset can be found at the link above.

3. If you wish to perform distillation, you can run:：
     `cd TrackRec`
     `bash run_distill.sh`
   
4. If you want to conduct iterative alternating feedback learning, you can run the following script (which includes all processes):
     `bash self_play_by_epoch`

5. To test the metrics of the iterative alternating feedback learning, you can run the following code:：
     `cd TrackRec/test`
     `bash test.sh`

6. TrackRec can integrate the generated RecCoT into any traditional recommendation model to improve its performance. We provide the RecCoT generated at each stage along with the corresponding embeddings, which can be found at the link above.
    For example, with Amazon-Books, you can run:
     `cd Rec/RS
     `python run_ctr.py` for ctr task

Our implementation code is based on : 
[safe-rlhf]([https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation/tree/main](https://github.com/PKU-Alignment/safe-rlhf))
[KAR](https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation/tree/main)
