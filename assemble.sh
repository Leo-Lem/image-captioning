DIR=${1:-"/workspaces/w24/bai/paper/src"}
FILE=$DIR/${2:-"out/__assembled__"}.py

# 'compile' all python files to the model.py file
if [ -f $FILE ]; then
    rm $FILE
fi

cat $DIR/data/vocabulary.py >> $FILE
cat $DIR/data/flickrdataset.py >> $FILE
cat $DIR/data/*.py >> $FILE
cat $DIR/models/*.py >> $FILE
cat $DIR/eval/*.py >> $FILE
cat $DIR/__main__.py >> $FILE

# remove all relative imports
sed -i 's/from \..* import .*//g' $FILE
sed -i 's/from data import .*//g' $FILE
sed -i 's/from models import .*//g' $FILE
sed -i 's/from eval import .*//g' $FILE

printf "Assembled python files to $FILE\n"