# RLaffinity
Contrastive 3D Convolution Neural Network for RNA and small molecule binding affinity prediction.

# Introduction
The diverse structures and functions inherent in RNAs present a wealth of potential drug targets. Hence some small molecules are anticipated to serve as leading compounds, providing guidance for the development of novel RNA-targeted therapeutics, distinct from conventional therapeutic approaches primarily focused on protein targets. Consequently, the determination of RNA-small molecule binding affinities is a critical undertaking in the landscape of RNA-targeted drug discovery and development. Nevertheless, to date, no computational method of RNA-small molecule binding affinity has been proposed. The prediction of RNA-small molecule binding affinities remains a significant challenge. The development of a computational model is deemed essential to effectively extract relevant features and predict RNA-small molecule binding affinities accurately. In this research paper, we introduce RLaffinity, a novel deep learning model designed for the prediction of RNA-small molecule binding affinities based on structure-derived properties. RLaffinity integrates information from RNA pockets and small molecules, utilizing a 3D convolutional neural network (3D-CNN) coupled with a contrastive learning-based self-supervised pre-training model. To the best of our knowledge, this study represents the inaugural attempt to employ a 3D-CNN for the prediction of RNA-small molecule binding affinities. Our experimental results showcase RLaffinity's superior performance compared to baseline methods in the realm of binding affinity prediction. The efficacy of RLaffinity underscores the capability of 3D-CNN to accurately extract both global pocket information and local neighbor nucleotide information within RNAs. Notably, the integration of a self-supervised pre-training model significantly enhances predictive performance. Additionally, RLaffinity also can be used as a potential tool for RNA-targeted drugs virtual screening.

# Requirements
1.anaconda
2. torch
3. numpy
4. pandas
5. scipy
6. tqdm
7. math
8. Bio

# Usage
Re-train models:
1.	Prepare data for pre-training.
run ‘process_pdbbind.py receptor_dir ligand_dir --out_dir output_dir’
run ‘prepare_lmdb.py input_file_path output_path’
2.	Re-train pre-training model.
run ‘trainstage1.py’
3.	Prepare data for training and testing.
run ‘process_pdbbind.py receptor_dir ligand_dir --out_dir output_dir’
run ‘prepare_lmdb.py input_file_path output_path -s --train_txt  data/train_list.txt --val_txt data/val_list.txt --test_txt data/test_list.txt --score_path data/input_label/pdbbind_NL_cleaned.csv’
4.	Re-train and test model.
Run ‘train.py --data_dir data --mode test --output_dir output_train’

Only do predictions:
1.	Prepare testing data.
run ‘process_pdbbind.py receptor_dir ligand_dir --out_dir output_dir’
run ‘prepare_lmdb.py input_file_path output_path’
2.	Test model.
Run ‘test.py --data_dir data --output_dir output_test’

# Contact
sunsaisai@xidian.edu.cn

