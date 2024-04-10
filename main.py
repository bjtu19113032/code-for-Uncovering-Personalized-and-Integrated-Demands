
from pretrain import PreTrainingPipeline
from jointtrain import JointTrainer
from patterns import PatternFinder
import pandas as pd
import numpy as np
from features import FeatureSelection
behavior_size=500
embedding_size=16
pre_train_epochs=5
joint_train_epochs=20
n_classes=10
#load your dataframe as df
BAE, history,df2,raw = PreTrainingPipeline(df, behavior_size=behavior_size, least_behavior=5,embedding_size=embedding_size, epochs=pre_train_epochs).run()
labels =JointTrainer(df=df, BAE=BAE, n_classes=n_classes, n_epochs=joint_train_epochs).run()
pattern_finder = PatternFinder(raw_data=raw, labels=labels, n_classes=n_classes)
pattern_finder.find_patterns()
patterns_by_class = pattern_finder.get_patterns_by_class()
for label, patterns in patterns_by_class.items():
    print(f"Class {label} unique patterns (after removing common patterns): {patterns}")
fs = FeatureSelection(BAE=BAE,embedding_size=embedding_size, behavior_size=behavior_size)
top_10_features = fs.run(df2, labels)