
DATA_NAME="sr_20_batch740000_formulas606136.pickle"
DATA_URL="https://mdumenci.s3.us-east-2.amazonaws.com/$DATA_NAME"
REPO_URL="https://github.com/mertdumenci/sat-ml.git"

# Download data, set up folders
wget "$DATA_URL"
mkdir checkpoints

# Set up environment
python3 -m venv venv && source venv/bin/activate
pip install torch pytorch-ignite numpy tqdm tensorboard future
git clone "$REPO_URL" && cd "sat-ml"

mkdir runs
tensorboard --logdir runs &

# python branch_prediction.py --model lstm --data data/toy_batch0_formulas2556.pickle  --checkpoint checkpoint/ --batch-size 5 --log-interval 1