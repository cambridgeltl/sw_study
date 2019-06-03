if [[ $# -ne 7 ]]; then
  echo "$0 lang subword word_list_file sw_file emb_dir dict_dir suffix"
  exit 1
fi

lang=$1
subword=$2
in_file=$3
sw_file=$4
emb_dir=$5
dict_dir=$6
suffix=$7

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

for compf in "${compfs[@]}"
do
  emb_file=$emb_dir/$affix.$subword.$compf.$suffix
  dict_file=$dict_dir/$affix.$subword.$compf.dict.pth
  echo "MODEL:"${emb_file}
  python3 gen_word_emb.py --lang $lang \
    --subword $subword \
    --compf $compf \
    --in_file $in_file \
    --sw_file $sw_file \
    --emb_model $emb_file \
    --dict_file $dict_file 
done
