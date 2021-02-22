GPU=$1
dataset_path=$2
tmp_path=$3

echo "Getting phrases.."
python3 util/write_to_file.py ${dataset_path}
cd AutoPhrase
./auto_phrase.sh
./phrasal_segmentation.sh
cd ..
cp AutoPhrase/models/DBLP/segmentation.txt ${tmp_path}/segmentation.txt
python3 util/parse_autophrase_output.py ${tmp_path} ${dataset_path}
python3 preprocess.py ${dataset_path} ${tmp_path}
