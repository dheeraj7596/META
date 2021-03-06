use_gpu=$1
GPU=$2
dataset_path=$3
tmp_path=$4
print_flag=$5

echo "Cloning AutoPhrase.."
git clone https://github.com/shangjingbo1226/AutoPhrase.git AutoPhrase
sed -i.bak 's/DEFAULT_TRAIN=${DATA_DIR}\/EN\/DBLP.txt/#DEFAULT_TRAIN=${DATA_DIR}\/EN\/DBLP.txt/' AutoPhrase/auto_phrase.sh
sed -i.bak 's/RAW_TRAIN=${RAW_TRAIN:- $DEFAULT_TRAIN}/RAW_TRAIN=${DATA_DIR}\/EN\/text.txt/' AutoPhrase/auto_phrase.sh
rm AutoPhrase/auto_phrase.sh.bak
sed -i.bak 's/TEXT_TO_SEG=${TEXT_TO_SEG:- ${DATA_DIR}\/EN\/DBLP.5K.txt}/TEXT_TO_SEG=${TEXT_TO_SEG:- ${DATA_DIR}\/EN\/text.txt}/' AutoPhrase/phrasal_segmentation.sh
rm AutoPhrase/phrasal_segmentation.sh.bak
echo "Getting phrases.."
python3 util/write_to_file.py ${dataset_path}
cd AutoPhrase
./auto_phrase.sh
./phrasal_segmentation.sh
cd ..
cp AutoPhrase/models/DBLP/segmentation.txt ${tmp_path}/segmentation.txt
python3 util/parse_autophrase_output.py ${dataset_path} ${tmp_path}
python3 preprocess.py ${dataset_path} ${tmp_path}
python3 train.py ${dataset_path} ${tmp_path} ${use_gpu} ${GPU} ${print_flag}
rm -rf AutoPhrase/