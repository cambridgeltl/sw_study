if [[ $# -ne 5 ]]; then
  echo "$0 lang subword train_file sw_file save_dir"
  exit 1
fi

lang=$1
subword=$2
train_file=$3
sw_file=$4
save_dir=$5

if [ $lang = "en" ];
then
  affix=$lang.03
elif [ $lang = "de" ];
then
  affix=$lang.035
else
  affix=$lang
fi

if [ $subword = "sms" ]; 
then
  compfs=("add" "wpadd" "ppadd" "mpadd"
         "wwadd" "wwwpadd" "wwppadd" "wwmpadd"
         "att" "wpatt" "ppatt" "mpatt"
         "wwatt" "wwwpatt" "wwppatt" "wwmpatt"
         "mtxatt" "wpmtxatt" "ppmtxatt" "mpmtxatt"
         "wwmtxatt" "wwwpmtxatt" "wwppmtxatt" "wwmpmtxatt")
else
  compfs=("add" "ppadd" "mpadd"
         "wwadd" "wwppadd" "wwmpadd"
         "att" "ppatt" "mpatt"
         "wwatt" "wwppatt" "wwmpatt"
         "mtxatt" "ppmtxatt" "mpmtxatt"
         "wwmtxatt" "wwppmtxatt" "wwmpmtxatt")
fi

cur_dir=`pwd`
for compf in "${compfs[@]}"
do
  echo "MODEL:" $subword, $compf
  out_file=$save_dir/$affix.$subword.$compf.dict.pth
  echo "OUTPUT:" $out_file
  python3 $cur_dir/../sw/swbase_input_data.py --train $train_file \
    --subword $subword \
    --compf $compf \
    --sw_file $sw_file \
    --save_model $out_file
done
