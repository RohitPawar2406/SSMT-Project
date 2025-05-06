# SSMT-Project

## install dependencies
python3.10
pip install datasets[audio],  transformers, accelerate,  evaluate,  tensorboard, sacrebleu, comet

## To Run Exp 2 -  Train Speech-To-Text translation model 
a) python3 new_train_script.py
b) python3 inference_new.py => To inference

## To Run Exp3 - Finetune Whisper Model
a) python3 whisper.py
b) python preprocess.py and then python all_metric_intfere.py => To inference
