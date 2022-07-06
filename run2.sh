#python train.py --model bert --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/bert_train.log
#python train.py --model ernie --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/ernie_train.log
#python train.py --model ernie_healthy --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/ernie_healthy_train.log
#python train.py --model albert --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/albert_train.log
#python train.py --model roberta --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/roberta_train.log
#python train.py --model bert_wwm --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/bert_wwm_train.log
#python train.py --model reformer --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/reformer_train.log

python train.py --model bert --dir_name gangwei \
    --problem_type multi_label_classification \
    --epochs 10 --batch_size 16 > data/gangwei/bert_train.log
python train.py --model ernie --dir_name gangwei \
    --problem_type multi_label_classification \
    --epochs 10 --batch_size 16 > data/gangwei/ernie_train.log
python train.py --model ernie_healthy --dir_name gangwei \
    --problem_type multi_label_classification \
    --epochs 10 --batch_size 16 > data/gangwei/ernie_healthy_train.log
python train.py --model albert --dir_name gangwei \
    --problem_type multi_label_classification \
    --epochs 10 --batch_size 16 > data/gangwei/albert_train.log
python train.py --model roberta --dir_name gangwei \
    --problem_type multi_label_classification \
    --epochs 10 --batch_size 16 > data/gangwei/roberta_train.log
python train.py --model bert_wwm --dir_name gangwei \
    --problem_type multi_label_classification \
    --epochs 10 --batch_size 16 > data/gangwei/bert_wwm_train.log
#python train.py --model ernie --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/ernie_train.log
#python train.py --model ernie_healthy --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/ernie_healthy_train.log
#python train.py --model albert --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/albert_train.log
#python train.py --model roberta --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/roberta_train.log
#python train.py --model bert_wwm --dir_name xinwen --epochs 20 --batch_size 64 > data/xinwen/bert_wwm_train.log
