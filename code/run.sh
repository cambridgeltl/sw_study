if [[ $# -ne 5 ]]; then
  echo "$0 lang subword compf lr bs"
  exit 1
fi

lang=$1
subword=$2
compf=$3
lr=$4
bs=$5

min_count=5
iter=15
thr=16

cur_dir=$PWD
data_path=$cur_dir/../toy_data/en/
model_path=$data_path
train_file=$lang.sent.1m

if [ $subword = "bpe" ];
then
  sw_file=$model_path/$lang.wiki.bpe.vs10000.model
elif [ $subword = "sms" ] || [ $subword = "morf" ];
then
  sw_file=$data_path/$train_file.$min_count.$subword
else
  sw_file="-"
fi

outs=$train_file."$subword".$compf.ep$iter.lr$lr.bs$bs

# ------------------------------
# Training
# ------------------------------
# single thread
#python3 -u sw2vec.py --train $data_path/$train_file \
# use multiple threads by default
python3 -u mul_sw2vec.py --train $data_path/$train_file \
  --subword $subword \
  --compf $compf \
  --sw_file $sw_file \
  --thread $thr \
  --iter $iter \
  --lr  $lr \
  --min_count $min_count \
  --batch_size $bs \
  --cuda \
  --output $outs.vec.txt \
  --save_model $outs.pth \
  > $outs.log 2>&1

# output dictionary pth model
dict_outs=$train_file."$subword".$compf.dict.pth
echo "MODEL:" $subword, $compf >> $outs.log
echo "OUTPUT:" $dict_outs >> $outs.log
python3 sw/swbase_input_data.py --train $data_path/$train_file \
    --subword $subword \
    --compf $compf \
    --sw_file $sw_file \
    --min_count $min_count \
    --save_model $dict_outs \
    >> $outs.log 2>&1
